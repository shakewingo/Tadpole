"""The benchmark freeze: everything that must not move between a freeze and the runs under it.

A test session makes its own freeze with `python -m agent_from_scratch.evals dataset --output DIR`,
from committed code, and every `bench --final` run is checked against DIR/manifest.json.

`python -m agent_from_scratch.evals freeze` rewrites evals/bench/manifest.json instead. That file
is the Stage 8 reference the tests hold the val and Stage 8 test tasks to; it is not a freeze a
test run is checked against. Rewrite it only for a deliberate change to those tasks or the protocol.
"""

import dataclasses
import json
from pathlib import Path
import random
import subprocess
from tempfile import TemporaryDirectory

from .run import specs
from ..verify import digest, snapshot
from ...config import (AgentLimits, BENCH_DECODING, BENCH_MODEL, BENCH_N_CTX, BENCH_SEEDS,
                       CHAT_TEMPLATE_PATH)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST_PATH = HERE / "manifest.json"


def _source_hashes() -> dict[str, str]:
    return {p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in ROOT.rglob("*")
           if p.is_file() and p.suffix in {".py", ".jinja"} and "__pycache__" not in p.parts
           and "tests" not in p.parts}


def _worktree_clean() -> bool:
    """No tracked file has uncommitted changes, so `git_revision` names the hashed sources."""
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                            text=True, capture_output=True, check=True).stdout
    return not status.strip()


TASK_CONTENT_FIELDS = ("prompt_sha256", "expect_sha256", "fault_sha256", "tools_sha256",
                       "workspace_sha256")


def task_drift(new: dict, old: dict) -> list[str]:
    """IDs of test tasks added, removed or changed since an earlier freeze; [] if the same.

    Only task content is compared. Code may change between freezes; the test tasks may not.
    """
    def test_tasks(manifest: dict) -> dict[str, tuple]:
        return {task["id"]: tuple(task[field] for field in TASK_CONTENT_FIELDS)
                for task in manifest["tasks"] if task["split"] == "test"}

    a, b = test_tasks(new), test_tasks(old)
    return sorted(task_id for task_id in a.keys() | b.keys() if a.get(task_id) != b.get(task_id))


def build_manifest() -> dict:
    """Everything a Stage 8 test-split run must match. Loads no model; cheap enough to call
    before every `--final` run."""
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                              capture_output=True, check=True).stdout.strip()
    tasks = []
    for skeleton, ctx in specs():
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            rng = random.Random(f"bench:{ctx.name}:{ctx.seed}:{ctx.condition}")
            task = skeleton.build(rng, workspace, ctx)
            fault = dataclasses.asdict(task.fault) if task.fault is not None else None
            tasks.append({"id": task.id, "skeleton": task.skeleton, "split": task.split,
                         "prompt_sha256": digest(task.prompt.encode()),
                         "expect_sha256": digest(json.dumps(dataclasses.asdict(task.expect),
                                                            sort_keys=True).encode()),
                         "fault_sha256": digest(json.dumps(fault, sort_keys=True).encode()),
                         "tools_sha256": digest(json.dumps(task.tools).encode()),
                         "workspace_sha256": digest(json.dumps(snapshot(workspace),
                                                               sort_keys=True).encode())})
            if skeleton.strict:
                tasks[-1]["fault_signal"] = list(task.fault_signal) if task.fault_signal is not None else None
                tasks[-1]["fault_path"] = task.fault_path
            if task.followups:
                tasks[-1]["followups_sha256"] = digest(json.dumps(task.followups).encode())
    return {"source_sha256": _source_hashes(),
           "system_prompt_sha256": digest((ROOT / "prompts/system.md").read_bytes()),
           "chat_template_sha256": digest(CHAT_TEMPLATE_PATH.read_bytes()),
           "model": BENCH_MODEL, "n_ctx": BENCH_N_CTX, "decoding": BENCH_DECODING,
           "seeds": list(BENCH_SEEDS),
           "limits": dataclasses.asdict(AgentLimits()), "memory": "off",
           "splits_sha256": digest((HERE / "splits.json").read_bytes()),
           "tasks": tasks, "verifier_sha256": digest((HERE / "verify.py").read_bytes()),
           "git_revision": revision, "worktree_clean": _worktree_clean()}


def save_manifest() -> None:
    MANIFEST_PATH.write_text(json.dumps(build_manifest(), indent=2, sort_keys=True) + "\n")


def manifest_drift(path: Path) -> dict[str, tuple | str]:
    """Fields that differ from the freeze at `path`; {} if it is missing or unchanged.

    `git_revision` is recorded for provenance but never compared: committing the frozen
    manifest itself always changes HEAD, so gating on it would make every freeze self-drift
    against its own commit the moment it lands.

    `worktree_clean` is not compared either. A freeze made with uncommitted changes is drift
    in itself, since its git revision does not name the code it hashed; the current tree's
    state alone is not, because any change to a hashed file already shows in its hash.
    A manifest from before the field existed is not refused for lacking it.
    """
    if not path.exists():
        return {}
    frozen = json.loads(path.read_text())
    if not isinstance(frozen, dict) or not isinstance(frozen.get("tasks"), list):
        raise ValueError("Expected a manifest object with a tasks list.")
    if any(not isinstance(task, dict) or not isinstance(task.get("id"), str)
           for task in frozen["tasks"]):
        raise ValueError("Each manifest task needs a string id.")
    if len({task["id"] for task in frozen["tasks"]}) != len(frozen["tasks"]):
        raise ValueError("Duplicate manifest task ids.")
    current = build_manifest()
    drift = {key: (frozen.get(key), current.get(key)) for key in frozen.keys() | current.keys()
            if key not in ("tasks", "git_revision", "worktree_clean")
            and frozen.get(key) != current.get(key)}
    if {t["id"]: t for t in frozen["tasks"]} != {t["id"]: t for t in current["tasks"]}:
        drift["tasks"] = "task set or content changed"
    if frozen.get("worktree_clean") is False:
        drift["worktree_clean"] = "frozen with uncommitted changes; freeze from a clean clone"
    return drift


def protocol_mismatch(settings: dict, seed: int) -> list[str]:
    """Ways a loaded backend differs from the frozen protocol; [] means a comparable run.

    The served revision is not visible through the OpenAI API, so it is pinned by the launch
    command in evals/README.md rather than checked here.
    """
    problems = []
    if settings.get("backend") != BENCH_MODEL["backend"]:
        problems.append(f"backend {settings.get('backend')!r} is not {BENCH_MODEL['backend']!r}")
    if settings.get("model_root") != BENCH_MODEL["id"]:
        problems.append(f"served model {settings.get('model_root')!r} is not {BENCH_MODEL['id']!r}")
    for key, value in BENCH_DECODING.items():
        if settings.get(key) != value:
            problems.append(f"{key}={settings.get(key)!r}, frozen {value!r}")
    if settings.get("n_ctx") != BENCH_N_CTX:
        problems.append(f"n_ctx={settings.get('n_ctx')!r}, frozen {BENCH_N_CTX}")
    if seed not in BENCH_SEEDS:
        problems.append(f"seed {seed} is not one of the frozen samples {list(BENCH_SEEDS)}")
    return problems


def main():
    save_manifest()
    print(f"Wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
