"""A todo list the harness shows back to the model before every request."""

from .base import Tool, ToolErrorCode, ToolExecutionError


class PlanTool(Tool):
    name = "update_plan"
    description = (
        "Create or replace your task plan (a todo list). The harness shows the current plan "
        "back to you before every request. For any task of about 3+ steps, call this FIRST. "
        "Pass the COMPLETE list every time; it replaces the previous one. At most one task may "
        "be in_progress: set a task in_progress before working on it. You can mark only that "
        "task completed, and only when the last tool call you made while it was in_progress "
        "succeeded; any other update is refused. Skip planning for a single trivial step or a "
        "purely informational request. Touches no files."
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
        # Whether the last other tool call since the current item became in_progress
        # succeeded; None until the first one. Status is self-report, this is evidence.
        self._last_ok: bool | None = None

    @staticmethod
    def _in_progress(plan: list[dict]) -> str | None:
        return next((task["content"] for task in plan if task["status"] == "in_progress"), None)

    def record_call(self, ok: bool) -> None:
        """Record the result of a tool call other than update_plan."""
        self._last_ok = ok

    def execute(self, plan: list[dict]) -> dict:
        active = [task["content"] for task in plan if task["status"] == "in_progress"]
        if len(active) > 1:
            raise ToolExecutionError(ToolErrorCode.DENIED,
                f"At most one item may be in_progress; this update has {len(active)}: "
                f"{', '.join(map(repr, active))}. The plan is unchanged.")
        current = self._in_progress(self.plan)
        completed = {task["content"] for task in self.plan if task["status"] == "completed"}
        for task in plan:
            content = task["content"]
            if task["status"] != "completed" or content in completed:
                continue  # Not a move to completed: resending a completed item is fine.
            if content != current:
                reason = "only the in_progress item can be completed; set it in_progress and do the work first"
            elif self._last_ok is None:
                reason = "no tool call since it became in_progress"
            elif not self._last_ok:
                reason = "the last tool call since it became in_progress failed"
            else:
                continue
            raise ToolExecutionError(ToolErrorCode.DENIED,
                f"{content!r} cannot be completed: {reason}. The plan is unchanged.")
        if self._in_progress(plan) != current:
            self._last_ok = None  # Evidence belongs to the item it was gathered for.
        self.plan = plan
        return {"tasks": len(plan), "in_progress": len(active)}

    @property
    def done(self) -> bool:
        """True when the plan has items and every one is completed."""
        return bool(self.plan) and all(task["status"] == "completed" for task in self.plan)

    def render(self) -> str:
        """Return the plan as one '[status] content' line per task."""
        return "\n".join(f"[{task['status']}] {task['content']}" for task in self.plan)
