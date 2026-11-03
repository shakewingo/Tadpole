"""Any OpenAI-compatible chat API as the agent's model: native tool calls in and out, the one
parser, an estimated context budget, retries, and the backend picked from the environment."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import httpx

from agent_from_scratch.agent import Agent
from agent_from_scratch.bots.telegram_bot import format_reply, handle_update
from agent_from_scratch.budget import ApiBudget, BudgetExceeded
from agent_from_scratch.config import interactive_limits
from agent_from_scratch.context import InstructionConfig, context_fits, window_share_ratio
from agent_from_scratch.llm import ChatAPIClient, LLM, ResponseError, ResponseType, make_model
from agent_from_scratch.session import SessionStore
from agent_from_scratch.tools.base import ToolRegistry
from agent_from_scratch.tools.files import ReadFileTool, WriteFileTool
from agent_from_scratch.tools.register import default_tool_schemas
from agent_from_scratch.trace import RunStopReason

MESSAGES = [{"role": "system", "content": "System rules."}, {"role": "user", "content": "Read a.txt."}]


def reply(content="", *calls, reasoning=None, finish=None, prompt_tokens=120):
    """One chat completion as an OpenAI-compatible API returns it; `calls` are (id, name, arguments)."""
    message = {"role": "assistant", "content": content,
               "tool_calls": [{"id": call_id, "type": "function",
                               "function": {"name": name, "arguments": json.dumps(arguments)}}
                              for call_id, name, arguments in calls] or None}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {"model": "some-model",
            "choices": [{"index": 0, "message": message,
                         "finish_reason": finish or ("tool_calls" if calls else "stop")}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 20,
                      "total_tokens": prompt_tokens + 20}}


class FakeAPI:
    """A scripted server: POST /v1/chat/completions, after `failures` 503s."""

    def __init__(self, replies=(), *, failures=0, status=503):
        self.replies, self.failures, self.status, self.sent, self.headers = list(replies), failures, status, [], []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.headers.append(request.headers)
        if self.failures:
            self.failures -= 1
            return httpx.Response(self.status, text="busy")
        assert request.url.path == "/v1/chat/completions", request.url.path
        self.sent.append(json.loads(request.content))
        return httpx.Response(200, json=self.replies.pop(0))


def client(server: FakeAPI, api_key: str | None = None, **kwargs) -> ChatAPIClient:
    return ChatAPIClient("https://api.test/v1", "some-model", api_key, backoff=0,
                         transport=httpx.MockTransport(server), **kwargs)


class GenerateTests(unittest.TestCase):
    def test_a_tool_call_reply_becomes_the_same_calls(self):
        server = FakeAPI([reply("Reading.", ("c1", "read_file", {"path": "a.txt"}),
                                ("c2", "read_file", {"path": "b.txt"}))])
        response = client(server).generate(MESSAGES, default_tool_schemas)
        self.assertEqual(response.type, ResponseType.tool_call)
        self.assertEqual([(c.name, c.arguments, c.call_id) for c in response.tool_calls],
                         [("read_file", {"path": "a.txt"}, "c1"), ("read_file", {"path": "b.txt"}, "c2")])
        self.assertEqual(response.content, "Reading.")
        self.assertEqual(response.usage, {"prompt_tokens": 120, "completion_tokens": 20, "total_tokens": 140})
        sent = server.sent[0]
        self.assertEqual(sent["model"], "some-model")
        self.assertEqual(sent["messages"], MESSAGES)
        self.assertEqual(sent["tools"], list(default_tool_schemas.values()))
        self.assertEqual((sent["temperature"], sent["stream"]), (0.0, False))

    def test_a_plain_reply_is_direct(self):
        response = client(FakeAPI([reply("Four.")])).generate(MESSAGES, default_tool_schemas)
        self.assertEqual((response.type, response.content, response.tool_calls), (ResponseType.direct, "Four.", []))

    def test_a_truncated_reply_is_the_parsers_error(self):
        with self.assertRaises(ResponseError) as caught:
            client(FakeAPI([reply("Half", finish="length")])).generate(MESSAGES, {})
        self.assertIsNotNone(caught.exception.raw_response)

    def test_the_key_is_sent_as_a_bearer_token(self):
        server = FakeAPI([reply("Hi."), reply("Hi.")])
        client(server, "sk-test").generate(MESSAGES, {})
        client(server).generate(MESSAGES, {})  # a local server needs no key
        self.assertEqual(server.headers[0]["authorization"], "Bearer sk-test")
        self.assertNotIn("authorization", server.headers[1])

    def test_reasoning_is_returned_with_its_tool_calls(self):
        """DeepSeek's thinking mode asks for a turn's reasoning back while tools are in play."""
        server = FakeAPI([reply("", ("c1", "read_file", {"path": "a.txt"}), reasoning="Read it first."),
                          reply("Done.")])
        model = client(server)
        first = model.generate(MESSAGES, default_tool_schemas)
        history = [*MESSAGES, first.to_message(),
                   {"role": "tool", "tool_call_id": "c1", "content": "{\"ok\": true}"}]
        model.generate(history, default_tool_schemas)
        self.assertEqual(server.sent[1]["messages"][2]["reasoning_content"], "Read it first.")
        self.assertNotIn("reasoning_content", server.sent[0]["messages"][1])


class RetryTests(unittest.TestCase):
    def test_a_server_error_is_retried(self):
        server = FakeAPI([reply("Fine.")], failures=2)
        self.assertEqual(client(server).generate(MESSAGES, {}).content, "Fine.")

    def test_a_rate_limit_is_retried(self):
        server = FakeAPI([reply("Fine.")], failures=1, status=429)
        self.assertEqual(client(server).generate(MESSAGES, {}).content, "Fine.")

    def test_persistent_errors_are_raised_as_a_backend_error(self):
        with self.assertRaises(RuntimeError) as caught:
            client(FakeAPI(failures=3)).generate(MESSAGES, {})
        self.assertIn("503", str(caught.exception))

    def test_a_client_error_is_not_retried(self):
        server = FakeAPI([reply("never")], failures=1, status=401)
        with self.assertRaises(httpx.HTTPStatusError):
            client(server).generate(MESSAGES, {})
        self.assertEqual(server.replies, [reply("never")])


class BudgetTests(unittest.TestCase):
    def test_the_budget_is_an_estimate_the_fit_rule_accepts(self):
        model = client(FakeAPI(), n_ctx=8000)
        measured = model.measure_context(MESSAGES, default_tool_schemas)
        self.assertEqual(measured["count_method"], "estimate")
        self.assertEqual((measured["window_tokens"], measured["response_reserve"]), (8000, 2048))
        self.assertEqual(measured["remaining_tokens"], 8000 - measured["prompt_tokens"] - 2048)
        self.assertTrue(context_fits(measured, 256))
        self.assertIsNotNone(window_share_ratio(measured))

    def test_the_estimate_rises_to_what_the_api_reports(self):
        """Reported prompt tokens above the estimate (dense text) raise every later estimate."""
        model = client(FakeAPI([reply("Hi.", prompt_tokens=100_000)]))
        before = model.measure_context(MESSAGES, {})["prompt_tokens"]
        model.generate(MESSAGES, {})
        self.assertGreater(model.measure_context(MESSAGES, {})["prompt_tokens"], before)

    def test_an_overfull_request_is_blocked(self):
        long = [{"role": "user", "content": "x" * 40_000}]
        self.assertFalse(context_fits(client(FakeAPI(), n_ctx=8000).measure_context(long, {}), 256))

    def test_settings_name_the_backend_and_model(self):
        settings = client(FakeAPI()).settings()
        self.assertEqual((settings["backend"], settings["model"], settings["base_url"]),
                         ("chat_api", "some-model", "https://api.test/v1"))
        self.assertNotIn("api_key", json.dumps(settings))


class SpendingTests(unittest.TestCase):
    """About 2100 tokens are reserved per MESSAGES request: its size plus the 2048 reply reserve."""

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name, "budget.json")

    def budget(self, requests: int = 100, tokens: int = 1_000_000) -> ApiBudget:
        return ApiBudget(self.path, max_requests_per_hour=requests, max_tokens_per_day=tokens)

    def test_every_retry_is_charged_and_the_limit_stops_the_retries(self):
        server = FakeAPI(failures=3)
        with self.assertRaises(BudgetExceeded):
            client(server, budget=self.budget(requests=2)).generate(MESSAGES, {})
        self.assertEqual(len(server.headers), 2)

    def test_a_restarted_process_sends_nothing_once_the_limit_is_reached(self):
        client(FakeAPI([reply("Fine.")]), budget=self.budget(requests=1)).generate(MESSAGES, {})
        server = FakeAPI([reply("never")])
        with self.assertRaises(BudgetExceeded):
            client(server, budget=self.budget(requests=1)).generate(MESSAGES, {})
        self.assertEqual(server.headers, [])

    def test_reported_usage_replaces_the_reservation(self):
        """Unsettled, the second request would need over 4000; settled, 140 plus its reservation."""
        model = client(FakeAPI([reply("One."), reply("Two.")]), budget=self.budget(tokens=2400))
        model.generate(MESSAGES, {})
        self.assertEqual(model.generate(MESSAGES, {}).content, "Two.")

    def test_the_whole_request_is_reserved_before_it_is_sent(self):
        long = [{"role": "user", "content": "x" * 30_000}]  # about 10000 tokens
        server = FakeAPI([reply("never")])
        with self.assertRaises(BudgetExceeded):
            client(server, budget=self.budget(tokens=9000)).generate(long, {})
        self.assertEqual(server.headers, [])

    def test_a_failed_reply_costs_no_tokens(self):
        for failure in (503, 429):
            with self.subTest(status=failure):
                self.path.unlink(missing_ok=True)
                server = FakeAPI([reply("Fine.")], failures=1, status=failure)
                model = client(server, budget=self.budget(tokens=2300))
                self.assertEqual(model.generate(MESSAGES, {}).content, "Fine.")

    def test_a_rejected_request_costs_no_tokens(self):
        budget = self.budget(tokens=2300)
        with self.assertRaises(httpx.HTTPStatusError):
            client(FakeAPI(failures=1, status=401), budget=budget).generate(MESSAGES, {})
        self.assertEqual(client(FakeAPI([reply("Fine.")]), budget=budget).generate(MESSAGES, {}).content, "Fine.")

    def test_a_timeout_keeps_its_reservation_since_the_model_may_have_run(self):
        attempts = []

        def timeout(request: httpx.Request) -> httpx.Response:
            attempts.append(request)
            raise httpx.ReadTimeout("slow", request=request)

        model = ChatAPIClient("https://api.test/v1", "some-model", backoff=0, transport=httpx.MockTransport(timeout),
                              budget=self.budget(tokens=2300))
        with self.assertRaises(BudgetExceeded):
            model.generate(MESSAGES, {})
        self.assertEqual(len(attempts), 1)

    def test_a_turn_stops_with_a_model_error_when_the_budget_runs_out(self):
        server = FakeAPI([reply("Reading.", ("c1", "read_file", {"path": "a.txt"})), reply("never")])
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "a.txt").write_text("alpha")
            agent = Agent(client(server, budget=self.budget(requests=1)),
                          registry=ToolRegistry([ReadFileTool(workspace)]),
                          instruction_config=InstructionConfig(workspace=workspace))
            result = agent.run_turn("Read a.txt.")
        self.assertEqual(result.stop_reason, RunStopReason.MODEL_ERROR)
        self.assertIn("BudgetExceeded", result.error_message)
        self.assertEqual(len(server.headers), 1)

    def test_the_bot_replies_with_the_reason_when_the_budget_is_spent(self):
        budget = self.budget(requests=1)
        budget.charge(1)
        server, telegram = FakeAPI([reply("never")]), Mock()
        update = {"update_id": 1, "message": {"from": {"id": 7}, "chat": {"id": 7}, "text": "Hello."}}
        with TemporaryDirectory() as directory:
            agent = Agent(client(server, budget=budget), directory, registry=ToolRegistry([]),
                          instruction_config=InstructionConfig(workspace=Path(directory)))
            handle_update(update, agent=agent, store=SessionStore(Path(directory, "sessions")), client=telegram,
                          allowed_user_id=7, active_sessions={}, compact_pending={})
        self.assertEqual(server.headers, [])
        (chat_id, text), _ = telegram.send_message.call_args
        self.assertEqual(chat_id, 7)
        self.assertIn("AGENT_MAX_REQUESTS_PER_HOUR", text)

    def test_a_refused_summary_reports_the_budget_when_the_request_cannot_fit_either(self):
        """The summary's input fits this window and the whole history does not, so without the
        summary the turn cannot continue; the reason must still be the budget."""
        budget = self.budget(requests=1)
        budget.charge(1)
        server = FakeAPI([reply("never")])
        history = []
        for index in range(6):
            history += [{"role": "user", "content": f"{index} " + "x" * 6000}, {"role": "assistant", "content": "ok"}]
        with TemporaryDirectory() as directory:
            model = client(server, budget=budget, n_ctx=14000)
            agent = Agent(model, registry=ToolRegistry([]),
                          instruction_config=InstructionConfig(workspace=Path(directory)))
            result = agent.run_turn("Continue.", history)
        self.assertFalse(context_fits(model.measure_context(result.messages, {}), 256))
        self.assertEqual(result.model_requests[0].purpose, "compact")
        self.assertEqual(server.headers, [])
        self.assertEqual(result.stop_reason, RunStopReason.MODEL_ERROR)
        self.assertIn("AGENT_MAX_REQUESTS_PER_HOUR", format_reply(result))

    def test_a_refused_summary_is_not_attempted_again(self):
        """A retry at the same cut cannot succeed, and each one would log another refusal."""
        budget = self.budget(requests=1)
        budget.charge(1)
        history = [{"role": "user", "content": "x" * 6000}, {"role": "assistant", "content": "ok"},
                   {"role": "user", "content": "y" * 6000}, {"role": "assistant", "content": "ok"}]
        with TemporaryDirectory() as directory:
            agent = Agent(client(FakeAPI(), budget=budget), registry=ToolRegistry([]),
                          instruction_config=InstructionConfig(workspace=Path(directory)))
            result = agent.run_turn("Continue.", history, compact=True)
        self.assertEqual(sum(request.purpose == "compact" for request in result.model_requests), 1)
        self.assertEqual(result.stop_reason, RunStopReason.MODEL_ERROR)

    def test_a_remote_api_gets_a_budget_from_the_environment(self):
        environment = {"AGENT_BASE_URL": "https://api.deepseek.com", "AGENT_MODEL": "deepseek-flash",
                       "AGENT_BUDGET_FILE": str(self.path), "AGENT_MAX_REQUESTS_PER_HOUR": "1"}
        with patch.dict("os.environ", environment, clear=True):
            model = make_model()
        self.assertEqual(model.budget.path, self.path)
        model.budget.charge(1)
        with self.assertRaises(BudgetExceeded):
            model.budget.charge(1)

    def test_a_remote_api_is_refused_a_token_limit_that_cannot_be_read(self):
        for variable in ("AGENT_MAX_REQUESTS_PER_HOUR", "AGENT_MAX_TOKENS_PER_DAY"):
            for value in ("0", "-1", "lots"):
                environment = {"AGENT_BASE_URL": "https://api.deepseek.com", "AGENT_MODEL": "deepseek-flash",
                               "AGENT_BUDGET_FILE": str(self.path), variable: value}
                with self.subTest(variable=variable, value=value), patch.dict("os.environ", environment, clear=True):
                    with self.assertRaises(ValueError):
                        make_model()

    def test_a_local_server_costs_nothing_and_has_no_budget(self):
        for host in ("localhost", "127.0.0.1", "[::1]"):
            environment = {"AGENT_BASE_URL": f"http://{host}:11434/v1", "AGENT_MODEL": "qwen3:4b"}
            with self.subTest(host=host), patch.dict("os.environ", environment, clear=True):
                self.assertIsNone(make_model().budget)


class MakeModelTests(unittest.TestCase):
    def test_a_base_url_picks_the_chat_api(self):
        environment = {"AGENT_BASE_URL": "http://localhost:11434/v1", "AGENT_MODEL": "qwen3:4b",
                       "AGENT_N_CTX": "4096"}
        with patch.dict("os.environ", environment, clear=True):
            model = make_model()
        self.assertIsInstance(model, ChatAPIClient)
        self.assertEqual((model.base_url, model.model, model.n_ctx), ("http://localhost:11434/v1", "qwen3:4b", 4096))

    def test_a_base_url_needs_a_model(self):
        with patch.dict("os.environ", {"AGENT_BASE_URL": "http://localhost:11434/v1"}, clear=True):
            with self.assertRaises(ValueError):
                make_model()

    def test_no_base_url_keeps_the_local_gguf(self):
        with patch.dict("os.environ", {}, clear=True), patch("agent_from_scratch.llm.LLM") as local:
            self.assertIs(make_model(), local.return_value)
        local.assert_called_once_with(lora_path=None)

    def test_a_lora_path_puts_an_adapter_on_the_local_gguf(self):
        with patch.dict("os.environ", {"AGENT_LORA_PATH": "models/ei-3-lora-f16.gguf"}, clear=True), \
                patch("agent_from_scratch.llm.LLM") as local:
            make_model()
        local.assert_called_once_with(lora_path="models/ei-3-lora-f16.gguf")

    def test_planning_is_on_for_the_terminal_and_bot_unless_an_adapter_is_loaded(self):
        lora = {"AGENT_LORA_PATH": "models/ei-3-lora-f16.gguf"}
        api = {"AGENT_BASE_URL": "https://api.deepseek.com"}
        for env, planning in (({}, True), (lora, False), ({**lora, **api}, True)):
            with self.subTest(env=env), patch.dict("os.environ", env, clear=True):
                limits = interactive_limits()
                self.assertEqual((limits.planning, limits.max_iterations, limits.max_tool_calls),
                                 (planning, 30, 60))


class AgentLoopTests(unittest.TestCase):
    def test_a_two_tool_task_runs_end_to_end(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "a.txt").write_text("42")
            server = FakeAPI([reply("", ("c1", "read_file", {"path": "a.txt"})),
                              reply("", ("c2", "write_file", {"path": "b.txt", "content": "84"})),
                              reply("Wrote 84 to b.txt.")])
            registry = ToolRegistry([ReadFileTool(workspace, restrict_to_workspace=True),
                                     WriteFileTool(workspace, restrict_to_workspace=True)])
            agent = Agent(client(server), str(root / "state"), registry=registry,
                          instruction_config=InstructionConfig(workspace=workspace))
            result = agent.run_turn("Double the number in a.txt into b.txt.")
            self.assertEqual(result.stop_reason, RunStopReason.FINAL_RESPONSE)
            self.assertEqual(result.final_answer, "Wrote 84 to b.txt.")
            self.assertEqual((workspace / "b.txt").read_text(), "84")
            self.assertEqual(result.settings["backend"], "chat_api")
            # The second request carries the first call and its result back to the API.
            second = server.sent[1]["messages"]
            self.assertEqual(second[-2]["tool_calls"][0]["id"], "c1")
            self.assertEqual((second[-1]["role"], second[-1]["tool_call_id"]), ("tool", "c1"))

    def test_reasoning_follows_a_repeated_call_id_the_agent_renames(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "a.txt").write_text("42")
            server = FakeAPI([reply("", ("call_0", "read_file", {"path": "a.txt"}), reasoning="first"),
                              reply("", ("call_0", "read_file", {"path": "a.txt"}), reasoning="second"),
                              reply("42.")])
            registry = ToolRegistry([ReadFileTool(workspace, restrict_to_workspace=True)])
            agent = Agent(client(server), str(root / "state"), registry=registry,
                          instruction_config=InstructionConfig(workspace=workspace))
            agent.run_turn("Read a.txt twice.")
            turns = [m for m in server.sent[2]["messages"] if m.get("tool_calls")]
            self.assertNotEqual(turns[0]["tool_calls"][0]["id"], turns[1]["tool_calls"][0]["id"])
            self.assertEqual([m["reasoning_content"] for m in turns], ["first", "second"])


if __name__ == "__main__":
    unittest.main()
