"""The paid-API ceiling: charged before a request leaves, kept across restarts, closed when unsure."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agent_from_scratch.budget import ApiBudget, BudgetExceeded

HOUR = 3600
DAY = 86400


class Clock:
    def __init__(self, now: float = 10 * DAY):
        self.now = now

    def __call__(self) -> float:
        return self.now


class ApiBudgetTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name, "nested", "budget.json")
        self.clock = Clock()

    def budget(self, requests: int = 100, tokens: int = 1000) -> ApiBudget:
        return ApiBudget(self.path, max_requests_per_hour=requests, max_tokens_per_day=tokens,
                         clock=self.clock)

    def test_requests_past_the_hourly_limit_are_refused(self):
        budget = self.budget(requests=2)
        budget.charge(1)
        budget.charge(1)
        with self.assertRaises(BudgetExceeded):
            budget.charge(1)

    def test_a_restart_does_not_refill_the_budget(self):
        """A crash loop builds a new process each time; its spending must still add up."""
        self.budget(requests=1).charge(1)
        with self.assertRaises(BudgetExceeded):
            self.budget(requests=1).charge(1)

    def test_the_request_count_reopens_in_the_next_hour(self):
        budget = self.budget(requests=1)
        budget.charge(1)
        self.clock.now += HOUR
        budget.charge(1)

    def test_tokens_past_the_daily_limit_are_refused(self):
        budget = self.budget(tokens=1000)
        budget.charge(600)
        with self.assertRaises(BudgetExceeded):
            budget.charge(600)

    def test_a_refused_charge_spends_nothing(self):
        budget = self.budget(tokens=1000)
        budget.charge(600)
        with self.assertRaises(BudgetExceeded):
            budget.charge(600)
        budget.charge(400)

    def test_the_token_count_survives_a_new_hour_and_reopens_the_next_day(self):
        budget = self.budget(tokens=1000)
        budget.charge(1000)
        self.clock.now += HOUR
        with self.assertRaises(BudgetExceeded):
            budget.charge(1)
        self.clock.now += DAY
        budget.charge(1000)

    def test_settling_replaces_the_reservation_with_what_was_used(self):
        budget = self.budget(tokens=1000)
        budget.charge(600)
        budget.settle(600, 100)
        budget.charge(900)
        with self.assertRaises(BudgetExceeded):
            budget.charge(1)

    def test_a_ledger_it_cannot_read_refuses_rather_than_starting_over(self):
        self.path.parent.mkdir(parents=True)
        for content in ("not json", "[]", '{"hour": 1}', '{"hour": 1, "requests": "0", "day": 1, "tokens": 0}'):
            with self.subTest(content=content):
                self.path.write_text(content)
                with self.assertRaises(BudgetExceeded):
                    self.budget().charge(1)


if __name__ == "__main__":
    unittest.main()
