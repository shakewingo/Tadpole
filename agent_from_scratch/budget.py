"""A ceiling on paid API use that survives restarts.

`AgentLimits` bounds one turn, but nothing there bounds how many turns run. A process that keeps
dying and replaying its work starts every counter over, so this one lives in a file, is shared
by every process on the machine, and is charged before a request leaves rather than after the
reply. It stops runaway use on this machine; a limit set with the provider is the only ceiling
on a key used anywhere else.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

from .config import API_BUDGET_PATH, MAX_API_REQUESTS_PER_HOUR, MAX_API_TOKENS_PER_DAY

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_FIELDS = ("hour", "requests", "day", "tokens")


class BudgetExceeded(RuntimeError):
    """A paid request was refused before it was sent."""


class ApiBudget:
    """Counts requests per UTC hour and tokens per UTC day in one JSON file.

    Args:
        path: the ledger; created on the first charge.
        max_requests_per_hour: every attempt counts, retries included.
        max_tokens_per_day: reported usage, or the reservation while none has been reported.
        clock: seconds since the epoch.
    """

    def __init__(self, path: str | Path, *, max_requests_per_hour: int, max_tokens_per_day: int,
                 clock: Callable[[], float] = time.time):
        self.path = Path(path)
        self.max_requests_per_hour = max_requests_per_hour
        self.max_tokens_per_day = max_tokens_per_day
        self.clock = clock

    def charge(self, tokens: int) -> None:
        """Reserve one request and `tokens`, or raise BudgetExceeded and reserve nothing."""
        with self._ledger() as ledger:
            if ledger["requests"] >= self.max_requests_per_hour:
                raise BudgetExceeded(
                    f"Paid API budget reached: {ledger['requests']} requests this hour "
                    f"(AGENT_MAX_REQUESTS_PER_HOUR={self.max_requests_per_hour}). Nothing was sent.")
            if ledger["tokens"] + tokens > self.max_tokens_per_day:
                raise BudgetExceeded(
                    f"Paid API budget reached: {ledger['tokens']} tokens used today, and this request "
                    f"could use {tokens} more (AGENT_MAX_TOKENS_PER_DAY={self.max_tokens_per_day}). "
                    "Nothing was sent.")
            ledger["requests"] += 1
            ledger["tokens"] += tokens

    def settle(self, charged: int, actual: int) -> None:
        """Replace a reservation with what the request really used."""
        with self._ledger() as ledger:
            ledger["tokens"] = max(0, ledger["tokens"] - charged) + actual

    @contextmanager
    def _ledger(self) -> Iterator[dict[str, int]]:
        """The current counts under an exclusive lock; saved unless the block raises."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_name(self.path.name + ".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ledger = self._read()
            now = self.clock()
            hour, day = int(now // 3600), int(now // 86400)
            if ledger["hour"] != hour:
                ledger["hour"], ledger["requests"] = hour, 0
            if ledger["day"] != day:
                ledger["day"], ledger["tokens"] = day, 0
            yield ledger
            temporary = self.path.with_name(self.path.name + ".tmp")
            temporary.write_text(json.dumps(ledger))
            os.replace(temporary, self.path)

    def _read(self) -> dict[str, int]:
        """A damaged ledger closes the budget: starting over would forget what was spent."""
        try:
            ledger = json.loads(self.path.read_text())
        except FileNotFoundError:
            return dict.fromkeys(_FIELDS, 0)
        except ValueError as error:
            raise BudgetExceeded(f"Paid API budget file {self.path} is unreadable: {error}. "
                                 "Nothing was sent; check or delete it.") from error
        if not isinstance(ledger, dict) or any(type(ledger.get(name)) is not int for name in _FIELDS):
            raise BudgetExceeded(f"Paid API budget file {self.path} is damaged. "
                                 "Nothing was sent; check or delete it.")
        return ledger


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer: {error}") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, not {value}.")
    return value


def budget_for(base_url: str, env: Mapping[str, str]) -> ApiBudget | None:
    """The budget a chat API needs: none for a server on this machine, which costs nothing."""
    if urlsplit(base_url).hostname in LOCAL_HOSTS:
        return None
    return ApiBudget(env.get("AGENT_BUDGET_FILE") or API_BUDGET_PATH,
                     max_requests_per_hour=_positive_int(env, "AGENT_MAX_REQUESTS_PER_HOUR",
                                                         MAX_API_REQUESTS_PER_HOUR),
                     max_tokens_per_day=_positive_int(env, "AGENT_MAX_TOKENS_PER_DAY",
                                                      MAX_API_TOKENS_PER_DAY))
