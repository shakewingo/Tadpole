"""A todo list the harness shows back to the model before every request."""

from .base import Tool


class PlanTool(Tool):
    name = "update_plan"
    description = (
        "Create or replace your task plan (a todo list). The harness shows the current plan "
        "back to you before every request. For any task of about 3+ steps, call this FIRST. "
        "Pass the COMPLETE list every time; it replaces the previous one. Keep exactly one "
        "task in_progress and mark a task completed as soon as it is done. Skip planning for a "
        "single trivial step or a purely informational request. Touches no files."
    )
    parameters = {
        "type": "object", "properties": {"plan": {"type": "array", "items": {
            "type": "object", "properties": {
                "content": {"type": "string"},
                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
            }, "required": ["content", "status"], "additionalProperties": False}}},
        "required": ["plan"], "additionalProperties": False,
    }

    def __init__(self):
        self.plan: list[dict] = []
        # Tool-call count when each item (by content) last appeared as work to
        # do. -1 marks an item carried over from an earlier turn: it may have
        # been worked on before this turn's planner existed, so completing it
        # is never flagged.
        self._appeared_at: dict[str, int] = {}
        self._status: dict[str, str] = {}
        self._tool_calls = 0

    @staticmethod
    def parse_rendered(rendered: str) -> list[dict]:
        """Parse render() output back into plan items.

        A rendered line is "[status] content"; a continuation line of a
        multi-line item follows its first line with no status prefix. A
        content line that itself starts with a status prefix is misread as a
        new item; that only costs a seed, never a crash.
        """
        items: list[dict] = []
        for line in rendered.split("\n"):
            status = next((candidate for candidate in ("pending", "in_progress", "completed")
                           if line.startswith(f"[{candidate}] ")), None)
            if status is not None:
                items.append({"content": line[len(status) + 3:], "status": status})
            elif items:
                items[-1]["content"] += "\n" + line
        return items

    def seed_existing(self, items: list[dict]) -> None:
        """Remember items from an earlier turn; completing them is never flagged."""
        for item in items:
            self._appeared_at[item["content"]] = -1
            self._status[item["content"]] = item["status"]
        self.plan = [dict(item) for item in items]

    def note_tool_call(self) -> None:
        """Record that a tool call other than update_plan has run."""
        self._tool_calls += 1

    def execute(self, plan: list[dict]) -> dict:
        warnings: list[str] = []
        for item in plan:
            content, status = item["content"], item["status"]
            if (content not in self._appeared_at
                    or (self._status.get(content) == "completed" and status != "completed")):
                # First seen now, or reopened as work to do: it appears now.
                self._appeared_at[content] = self._tool_calls
            self._status[content] = status
            if status == "completed" and self._appeared_at[content] == self._tool_calls:
                warnings.append(content)
        self.plan = plan
        result = {"tasks": len(plan),
                  "in_progress": sum(task["status"] == "in_progress" for task in plan)}
        if warnings:
            result["warning"] = (
                "Marked completed with no tool call since the item appeared: "
                + "; ".join(warnings)
                + ". Confirm the work actually ran before reporting it done."
            )
        return result

    @property
    def done(self) -> bool:
        """True when the plan has items and every one is completed."""
        return bool(self.plan) and all(task["status"] == "completed" for task in self.plan)

    def render(self) -> str:
        """Return the plan as one '[status] content' line per task."""
        return "\n".join(f"[{task['status']}] {task['content']}" for task in self.plan)
