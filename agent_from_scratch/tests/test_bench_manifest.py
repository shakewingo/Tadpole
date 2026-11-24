"""The freeze manifest is reproducible and reports drift when a frozen file changes."""

import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from agent_from_scratch.evals.bench.manifest import ROOT, build_manifest, manifest_drift, task_drift


class ManifestTests(unittest.TestCase):
    def frozen_file(self, content: dict) -> Path:
        directory = self.enterContext(TemporaryDirectory())
        path = Path(directory) / "manifest.json"
        path.write_text(json.dumps(content))
        return path

    def test_build_manifest_is_reproducible_and_has_no_model_dependent_fields(self):
        first, second = build_manifest(), build_manifest()
        self.assertEqual(first, second)
        self.assertEqual(first["memory"], "off")
        # The Stage 8-9 tasks alone; the Stage 10 shapes join at the levels levels.json names.
        with patch("agent_from_scratch.evals.bench.skeletons.load_levels", return_value={}):
            stage9 = build_manifest()
        counts = {split: sum(task["split"] == split for task in stage9["tasks"])
                  for split in ("train", "val", "test")}
        self.assertEqual(counts, {"train": 420, "val": 104, "test": 140})
        self.assertLessEqual({t["id"] for t in stage9["tasks"]}, {t["id"] for t in first["tasks"]})

    def test_manifest_drift_is_empty_for_a_missing_file(self):
        self.assertEqual(manifest_drift(Path("/nonexistent.json")), {})

    def test_manifest_drift_reports_a_changed_field(self):
        frozen = build_manifest()
        frozen["memory"] = "on"  # simulate a manifest that no longer matches the tree
        self.assertIn("memory", manifest_drift(self.frozen_file(frozen)))

    def test_manifest_drift_never_gates_on_git_revision(self):
        # Regression: committing the frozen manifest always changes HEAD, so a manifest that
        # only differs by git_revision (exactly what happens the instant `freeze` is committed)
        # must not be reported as drift.
        frozen = build_manifest()
        frozen["git_revision"] = "0" * 40
        self.assertNotIn("git_revision", manifest_drift(self.frozen_file(frozen)))


class WorktreeStateTests(unittest.TestCase):
    """A freeze hashes the working tree, so only one made from committed code names its sources."""

    def frozen_file(self, content: dict) -> Path:
        directory = self.enterContext(TemporaryDirectory())
        path = Path(directory) / "manifest.json"
        path.write_text(json.dumps(content))
        return path

    def test_build_manifest_records_whether_tracked_files_have_uncommitted_changes(self):
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                text=True, capture_output=True, check=True).stdout
        self.assertEqual(build_manifest()["worktree_clean"], status == "")

    def test_a_freeze_from_a_dirty_tree_is_drift(self):
        with patch("agent_from_scratch.evals.bench.manifest._worktree_clean", return_value=False):
            frozen = build_manifest()
        with patch("agent_from_scratch.evals.bench.manifest._worktree_clean", return_value=True):
            drift = manifest_drift(self.frozen_file(frozen))
        self.assertIn("worktree_clean", drift)

    def test_the_current_tree_state_alone_is_not_drift(self):
        # Editing a README after a clean freeze changes no hashed source; it must not block a run.
        with patch("agent_from_scratch.evals.bench.manifest._worktree_clean", return_value=True):
            frozen = build_manifest()
        with patch("agent_from_scratch.evals.bench.manifest._worktree_clean", return_value=False):
            drift = manifest_drift(self.frozen_file(frozen))
        self.assertNotIn("worktree_clean", drift)

    def test_a_manifest_from_before_the_field_existed_is_not_refused_for_it(self):
        frozen = build_manifest()
        del frozen["worktree_clean"]
        self.assertNotIn("worktree_clean", manifest_drift(self.frozen_file(frozen)))


class TaskDriftTests(unittest.TestCase):
    """A new freeze must keep an earlier freeze's test tasks exactly, whatever the code did."""

    @staticmethod
    def task(task_id: str, split: str = "test", prompt: str = "p") -> dict:
        return {"id": task_id, "split": split, "prompt_sha256": prompt, "expect_sha256": "e",
                "fault_sha256": "f", "tools_sha256": "t", "workspace_sha256": "w"}

    def test_identical_test_tasks_have_no_drift(self):
        old = {"tasks": [self.task("a"), self.task("b")]}
        new = {"tasks": [self.task("b"), self.task("a")], "source_sha256": {"changed.py": "x"}}
        self.assertEqual(task_drift(new, old), [])

    def test_a_changed_test_task_is_reported(self):
        old = {"tasks": [self.task("a"), self.task("b")]}
        new = {"tasks": [self.task("a"), self.task("b", prompt="other")]}
        self.assertEqual(task_drift(new, old), ["b"])

    def test_an_added_or_missing_test_task_is_reported(self):
        old = {"tasks": [self.task("a"), self.task("b")]}
        new = {"tasks": [self.task("a"), self.task("c")]}
        self.assertEqual(task_drift(new, old), ["b", "c"])

    def test_other_splits_are_not_compared(self):
        old = {"tasks": [self.task("a"), self.task("t", split="train")]}
        new = {"tasks": [self.task("a"), self.task("t", split="train", prompt="other"),
                         self.task("h", split="hardtest")]}
        self.assertEqual(task_drift(new, old), [])


if __name__ == "__main__":
    unittest.main()
