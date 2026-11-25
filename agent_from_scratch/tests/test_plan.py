"""The plan is re-injected every request and never enters raw history."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from agent_from_scratch.agent import Agent
from agent_from_scratch.config import AgentLimits
from agent_from_scratch.context import ContextState
from agent_from_scratch.llm import LLM, LLMResponse, ResponseType
from agent_from_scratch.tests.test_turn import answer, call
from agent_from_scratch.tools.base import ToolCall, ToolResult
from agent_from_scratch.tools.plan import PlanTool


def plan_call(status="in_progress"):
    return LLM.parse_response({"choices": [{"message": {"role": "assistant", "content":
        '<tool_call>{"name": "update_plan", "arguments": {"plan": '
        f'[{{"content": "Add numbers", "status": "{status}"}}]}}}}</tool_call>'}}]})


class PlanTests(unittest.TestCase):
    def test_tool_replaces_the_whole_plan_and_renders_it(self):
        tool = PlanTool()
        result = tool.invoke({"plan": [{"content": "Read config", "status": "in_progress"},
                                       {"content": "Report", "status": "pending"}]})
        self.assertTrue(result.ok)
        self.assertEqual(tool.render(), "[in_progress] Read config\n[pending] Report")
        self.assertFalse(tool.invoke({"plan": [{"content": "x", "status": "done"}]}).ok)

    def test_view_carries_one_trailing_reminder_and_raw_is_unchanged(self):
        raw = [{"role": "system", "content": "rules"}, {"role": "user", "content": "go"}]
        state = ContextState(list(raw), turn_start=1, last_sent=1)
        self.assertEqual(state.messages(), raw)
        state.plan = ""
        self.assertIn("not created a plan", state.messages()[-1]["content"])
        state.plan = "[pending] Report"
        self.assertEqual(state.messages()[-1]["role"], "system")
        self.assertIn("[pending] Report", state.messages()[-1]["content"])
        self.assertEqual(len(state.messages()), 3)
        self.assertEqual(state.raw, raw)

    def test_tool_is_done_only_when_every_item_is_completed(self):
        tool = PlanTool()
        self.assertFalse(tool.done)
        for statuses, done in [(["completed", "completed"], True),
                               (["completed", "pending"], False),
                               (["in_progress", "completed"], False)]:
            tool.invoke({"plan": [{"content": f"task {i}", "status": s}
                                  for i, s in enumerate(statuses)]})
            self.assertEqual(tool.done, done, statuses)
        tool.invoke({"plan": [{"content": "Read config\nthen report", "status": "completed"}]})
        self.assertTrue(tool.done)

    def test_parse_rendered_round_trips_including_multiline_content(self):
        tool = PlanTool()
        tool.invoke({"plan": [{"content": "Read config\nthen report", "status": "completed"},
                              {"content": "Report", "status": "pending"}]})
        self.assertEqual(PlanTool.parse_rendered(tool.render()),
                         [{"content": "Read config\nthen report", "status": "completed"},
                          {"content": "Report", "status": "pending"}])
        self.assertEqual(PlanTool.parse_rendered(""), [])

    def test_completed_item_with_no_tool_call_since_it_appeared_is_flagged(self):
        tool = PlanTool()
        # Appears already completed: no tool call could have done the work.
        result = tool.invoke({"plan": [{"content": "Add numbers", "status": "completed"}]})
        self.assertTrue(result.ok)
        self.assertIn("no tool call since", result.output["warning"])
        self.assertIn("Add numbers", result.output["warning"])
        # Appears pending, then completed with still no tool call: flagged too.
        tool = PlanTool()
        tool.invoke({"plan": [{"content": "Add numbers", "status": "pending"}]})
        result = tool.invoke({"plan": [{"content": "Add numbers", "status": "completed"}]})
        self.assertIn("no tool call since", result.output["warning"])

    def test_completed_item_after_a_tool_call_is_not_flagged(self):
        tool = PlanTool()
        tool.invoke({"plan": [{"content": "Add numbers", "status": "in_progress"}]})
        tool.note_tool_call()
        result = tool.invoke({"plan": [{"content": "Add numbers", "status": "completed"}]})
        self.assertTrue(result.ok)
        self.assertNotIn("warning", result.output)
        # A tool call that ran before the item appeared is no evidence for it.
        tool = PlanTool()
        tool.note_tool_call()
        result = tool.invoke({"plan": [{"content": "Add numbers", "status": "completed"}]})
        self.assertIn("no tool call since", result.output["warning"])

    def test_carried_over_item_is_not_flagged(self):
        tool = PlanTool()
        tool.seed_existing(PlanTool.parse_rendered("[pending] Add numbers"))
        result = tool.invoke({"plan": [{"content": "Add numbers", "status": "completed"}]})
        self.assertTrue(result.ok)
        self.assertNotIn("warning", result.output)

    def test_reopened_item_is_flagged_again(self):
        tool = PlanTool()
        tool.invoke({"plan": [{"content": "Add numbers", "status": "completed"}]})
        tool.invoke({"plan": [{"content": "Add numbers", "status": "pending"}]})
        result = tool.invoke({"plan": [{"content": "Add numbers", "status": "completed"}]})
        self.assertIn("no tool call since", result.output["warning"])

    def test_completed_plan_drops_the_reminder(self):
        raw = [{"role": "system", "content": "rules"}, {"role": "user", "content": "go"}]
        state = ContextState(list(raw), turn_start=1, last_sent=1,
                             plan="[completed] Read config\n[completed] Report", plan_done=True)
        self.assertEqual(state.messages(), raw)

    def test_plan_with_remaining_items_keeps_the_reminder(self):
        raw = [{"role": "system", "content": "rules"}, {"role": "user", "content": "go"}]
        state = ContextState(list(raw), turn_start=1, last_sent=1,
                             plan="[completed] Read config\n[pending] Report")
        self.assertIn("[Current plan", state.messages()[-1]["content"])
        self.assertIn("[pending] Report", state.messages()[-1]["content"])


class PlanTurnTests(unittest.TestCase):
    """Own copy of the TurnTests fixture; subclassing would rerun every inherited test here."""

    def setUp(self):
        self.model = Mock(spec=LLM)
        self.model.measure_context.return_value = {
            "count_method": "exact", "prompt_tokens": 100, "window_tokens": 8000,
            "response_reserve": 512, "remaining_tokens": 7388,
        }
        self.model.settings.return_value = {}
        self.model.read_usage.side_effect = LLM.read_usage
        self.agent = Agent(self.model)
        self.seen = []
        prompt_dir = Path(self.enterContext(TemporaryDirectory()))
        (prompt_dir / "system.md").write_text("System", encoding="utf-8")
        self.enterContext(patch("agent_from_scratch.context.PROMPTS_DIR", prompt_dir))

    def script(self, *responses):
        responses = iter(responses)

        def generate(messages, tools, **kwargs):
            self.seen.append(deepcopy(messages))
            return next(responses)

        self.model.generate.side_effect = generate

    def test_planning_turn_shows_the_plan_and_never_dispatches_it_to_the_registry(self):
        self.agent.limits = replace(AgentLimits(), planning=True)
        self.script(plan_call(), answer("4"))
        with patch.object(self.agent, "execute_tool") as execute:
            result = self.agent.run_turn("2+2")
        execute.assert_not_called()
        self.assertIn("update_plan", result.model_requests[0].tools or {})
        self.assertIn("not created a plan", self.seen[0][-1]["content"])
        self.assertIn("[in_progress] Add numbers", self.seen[1][-1]["content"])
        self.assertEqual(sum(m["role"] == "system" for m in self.seen[1]), 2)
        self.assertFalse(any(m["role"] == "system" for m in result.messages[1:]))
        self.assertEqual((result.stop_reason, result.final_answer), ("final_response", "4"))

    def test_completion_claimed_before_any_work_is_warned(self):
        self.agent.limits = replace(AgentLimits(), planning=True)
        self.script(plan_call("completed"), answer("4"))
        result = self.agent.run_turn("2+2")
        tool_messages = [m for m in result.messages if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("no tool call since", tool_messages[0]["content"])
        self.assertIn("Add numbers", tool_messages[0]["content"])
        self.assertEqual((result.stop_reason, result.final_answer), ("final_response", "4"))

    def test_completion_after_a_tool_call_is_not_warned(self):
        self.agent.limits = replace(AgentLimits(), planning=True)
        self.script(plan_call(), call(), plan_call("completed"), answer("4"))
        with patch.object(self.agent, "execute_tool",
                          return_value=ToolResult(call_id="", tool_name="calculator",
                                                  ok=True, output={})) as execute:
            result = self.agent.run_turn("2+2")
        execute.assert_called_once()
        tool_messages = [m for m in result.messages if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), 3)
        self.assertNotIn("no tool call since", tool_messages[2]["content"])
        self.assertEqual((result.stop_reason, result.final_answer), ("final_response", "4"))

    def test_completion_in_the_same_response_as_the_work_call_is_warned(self):
        # The incident shape from issue #3: update_plan comes first in the
        # batch, so no tool call has run when the item is marked completed.
        self.agent.limits = replace(AgentLimits(), planning=True)
        batch = LLMResponse(role="assistant", content="", type=ResponseType.tool_call,
                            tool_calls=[ToolCall("update_plan",
                                                 {"plan": [{"content": "Add numbers",
                                                            "status": "completed"}]}, "c1"),
                                        ToolCall("calculator",
                                                 {"operation": "add", "left": 2, "right": 2}, "c2")])
        self.script(batch, answer("4"))
        with patch.object(self.agent, "execute_tool",
                          return_value=ToolResult(call_id="", tool_name="calculator",
                                                  ok=True, output={})):
            result = self.agent.run_turn("2+2")
        tool_messages = [m for m in result.messages if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), 2)
        self.assertIn("no tool call since", tool_messages[0]["content"])

    def test_planning_off_adds_no_tool_or_reminder(self):
        self.script(answer("4"))
        result = self.agent.run_turn("2+2")
        self.assertNotIn("update_plan", result.model_requests[0].tools or {})
        self.assertEqual(self.seen[0][-1]["role"], "user")


if __name__ == "__main__":
    unittest.main()
