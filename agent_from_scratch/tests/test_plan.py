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
from agent_from_scratch.tools.base import ToolCall, ToolErrorCode
from agent_from_scratch.tools.plan import PlanTool

CALCULATOR = ("calculator", {"operation": "add", "left": 2, "right": 2})


def plan_call(status="in_progress"):
    return LLM.parse_response({"choices": [{"message": {"role": "assistant", "content":
        '<tool_call>{"name": "update_plan", "arguments": {"plan": '
        f'[{{"content": "Add numbers", "status": "{status}"}}]}}}}</tool_call>'}}]})


def plan_args(status):
    return ("update_plan", {"plan": [{"content": "Add numbers", "status": status}]})


def batch(*calls):
    """One response carrying several tool calls, run in order."""
    return LLMResponse(role="assistant", content="", type=ResponseType.tool_call,
                       tool_calls=[ToolCall(name, arguments) for name, arguments in calls])


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
            # The property only; how items may reach completed is tested below.
            tool.plan = [{"content": f"task {i}", "status": s} for i, s in enumerate(statuses)]
            self.assertEqual(tool.done, done, statuses)

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


def update(tool, *items):
    return tool.invoke({"plan": [{"content": content, "status": status} for content, status in items]})


class PlanCompletionTests(unittest.TestCase):
    """Status is self-report; only the in_progress item, after a successful call, completes (issue #3)."""

    def assertRefused(self, result, reason, tool, rendered):
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.DENIED)
        self.assertIn(reason, result.error_message)
        self.assertEqual(tool.render(), rendered)  # unchanged

    def test_completion_after_a_successful_call_is_accepted(self):
        tool = PlanTool()
        update(tool, ("Add numbers", "in_progress"))
        tool.record_call(True)
        self.assertTrue(update(tool, ("Add numbers", "completed")).ok)
        self.assertTrue(tool.done)

    def test_completion_with_no_call_yet_is_refused(self):
        tool = PlanTool()
        update(tool, ("Add numbers", "in_progress"))
        self.assertRefused(update(tool, ("Add numbers", "completed")),
                           "no tool call since it became in_progress", tool, "[in_progress] Add numbers")
        self.assertFalse(tool.done)

    def test_pending_item_cannot_jump_to_completed(self):
        tool = PlanTool()
        update(tool, ("Add numbers", "pending"))
        tool.record_call(True)
        self.assertRefused(update(tool, ("Add numbers", "completed")),
                           "only the in_progress item", tool, "[pending] Add numbers")
        tool = PlanTool()
        self.assertRefused(update(tool, ("Add numbers", "completed")), "only the in_progress item", tool, "")

    def test_one_call_completes_only_the_item_in_progress(self):
        tool = PlanTool()
        update(tool, ("A", "in_progress"), ("B", "pending"), ("C", "pending"))
        tool.record_call(True)
        self.assertRefused(update(tool, ("A", "completed"), ("B", "completed"), ("C", "completed")),
                           "'B' cannot be completed", tool, "[in_progress] A\n[pending] B\n[pending] C")
        self.assertTrue(update(tool, ("A", "completed"), ("B", "pending"), ("C", "pending")).ok)

    def test_last_call_failed_blocks_completion(self):
        tool = PlanTool()
        update(tool, ("Run tests", "in_progress"))
        tool.record_call(True)
        tool.record_call(False)  # e.g. a shell command that exited non-zero
        self.assertRefused(update(tool, ("Run tests", "completed")),
                           "the last tool call since it became in_progress failed",
                           tool, "[in_progress] Run tests")
        tool.record_call(True)
        self.assertTrue(update(tool, ("Run tests", "completed")).ok)

    def test_switching_items_clears_the_evidence(self):
        tool = PlanTool()
        update(tool, ("A", "in_progress"), ("B", "pending"))
        tool.record_call(True)
        update(tool, ("A", "pending"), ("B", "in_progress"))
        self.assertRefused(update(tool, ("A", "pending"), ("B", "completed")),
                           "no tool call", tool, "[pending] A\n[in_progress] B")
        # Completing one item and starting the next in one update is the normal step.
        tool = PlanTool()
        update(tool, ("A", "in_progress"), ("B", "pending"))
        tool.record_call(True)
        self.assertTrue(update(tool, ("A", "completed"), ("B", "in_progress")).ok)
        self.assertRefused(update(tool, ("A", "completed"), ("B", "completed")),
                           "'B' cannot be completed: no tool call", tool, "[completed] A\n[in_progress] B")

    def test_resending_a_completed_item_is_accepted(self):
        tool = PlanTool()
        update(tool, ("A", "in_progress"), ("B", "pending"))
        tool.record_call(True)
        update(tool, ("A", "completed"), ("B", "pending"))
        self.assertTrue(update(tool, ("A", "completed"), ("B", "in_progress")).ok)
        self.assertTrue(update(tool, ("A", "completed"), ("B", "in_progress")).ok)

    def test_reopened_item_needs_a_new_successful_call(self):
        tool = PlanTool()
        update(tool, ("A", "in_progress"))
        tool.record_call(True)
        update(tool, ("A", "completed"))
        update(tool, ("A", "in_progress"))
        self.assertRefused(update(tool, ("A", "completed")), "no tool call", tool, "[in_progress] A")
        tool.record_call(True)
        self.assertTrue(update(tool, ("A", "completed")).ok)

    def test_two_items_in_progress_are_refused(self):
        tool = PlanTool()
        update(tool, ("A", "pending"), ("B", "pending"))
        self.assertRefused(update(tool, ("A", "in_progress"), ("B", "in_progress")),
                           "At most one item may be in_progress", tool, "[pending] A\n[pending] B")


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

    def test_completion_before_the_work_call_is_refused(self):
        # The incident from issue #3: update_plan comes first in the batch.
        self.agent.limits = replace(AgentLimits(), planning=True)
        self.script(batch(plan_args("completed"), CALCULATOR), answer("4"))
        result = self.agent.run_turn("2+2")
        tool_messages = [m for m in result.messages if m["role"] == "tool"]
        self.assertIn("cannot be completed", tool_messages[0]["content"])
        self.assertIn("skipped", tool_messages[1]["content"])  # the batch stops at a failure
        self.assertIn("not created a plan", self.seen[1][-1]["content"])
        self.assertEqual((result.stop_reason, result.final_answer), ("final_response", "4"))

    def test_start_work_and_complete_in_one_response_is_accepted(self):
        self.agent.limits = replace(AgentLimits(), planning=True)
        self.script(batch(plan_args("in_progress"), CALCULATOR, plan_args("completed")), answer("4"))
        result = self.agent.run_turn("2+2")
        tool_messages = [m for m in result.messages if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), 3)
        self.assertTrue(all('"ok": true' in m["content"] for m in tool_messages))
        self.assertEqual(self.seen[1][-1]["role"], "tool")  # completed plan: no reminder (#2)
        self.assertEqual((result.stop_reason, result.final_answer), ("final_response", "4"))

    def test_completion_after_a_failed_call_is_refused(self):
        self.agent.limits = replace(AgentLimits(), planning=True)
        self.script(plan_call(), call(right=0, operation="divide"), plan_call("completed"), answer("4"))
        result = self.agent.run_turn("2+2")
        tool_messages = [m for m in result.messages if m["role"] == "tool"]
        self.assertIn("last tool call since it became in_progress failed", tool_messages[2]["content"])
        self.assertIn("[in_progress] Add numbers", self.seen[3][-1]["content"])  # reminder stays
        self.assertEqual((result.stop_reason, result.final_answer), ("final_response", "4"))

    def test_planning_off_adds_no_tool_or_reminder(self):
        self.script(answer("4"))
        result = self.agent.run_turn("2+2")
        self.assertNotIn("update_plan", result.model_requests[0].tools or {})
        self.assertEqual(self.seen[0][-1]["role"], "user")


if __name__ == "__main__":
    unittest.main()
