"""Public generation and scripted-check entry points require no model/API."""

import json
from pathlib import Path
import subprocess
import sys
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agent_from_scratch.evals.bench import run
from agent_from_scratch.evals.bench.manifest import build_manifest
from agent_from_scratch.config import BENCH_DECODING, BENCH_MODEL, BENCH_N_CTX
from agent_from_scratch.tests.test_bench_skeletons import _model


class DatasetCommandTests(unittest.TestCase):
    def command(self, *args):
        return subprocess.run([sys.executable, "-m", "agent_from_scratch.evals", "dataset", *map(str, args)],
                              capture_output=True, text=True)

    def test_generates_only_requested_split_with_private_contracts(self):
        with TemporaryDirectory() as temporary:
            out = Path(temporary) / "data"
            result = self.command("--split", "train", "--output", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads((out / "metadata.json").read_text())
            # 420 Stage 8-9 train tasks, plus the Stage 10 shapes at the levels levels.json names.
            train = len(run.specs("train"))
            self.assertGreaterEqual(train, 420)
            self.assertEqual(metadata["task_counts"], {"train": train})
            tasks = list((out / "tasks").glob("*/task.json"))
            self.assertEqual(len(tasks), train)
            for path in tasks:
                task = json.loads(path.read_text())
                self.assertEqual(task["split"], "train")
                self.assertNotIn("task.json", [p.name for p in (path.parent / "workspace").iterdir()])

    def test_generates_only_the_val_split(self):
        with TemporaryDirectory() as temporary:
            out = Path(temporary) / "data"
            result = self.command("--split", "val", "--output", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads((out / "metadata.json").read_text())
            # 144 validation tasks; the default (train + test) is unchanged.
            val = len(run.specs("val"))
            self.assertGreaterEqual(val, 144)
            self.assertEqual(metadata["task_counts"], {"val": val})
            tasks = list((out / "tasks").glob("*/task.json"))
            self.assertEqual(len(tasks), val)
            for path in tasks:
                task = json.loads(path.read_text())
                self.assertEqual(task["split"], "val")

    def test_scripted_check_gates_the_hardtest_split(self):
        with TemporaryDirectory() as temporary:
            out = Path(temporary) / "data"
            result = self.command("--split", "hardtest", "--output", out, "--check")
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((out / "summary.json").read_text())
            # 80 hard-test tasks, gated the same way as the default train + test run.
            hardtest = len(run.specs("hardtest"))
            self.assertGreaterEqual(hardtest, 80)
            self.assertEqual(summary["correct"], {"passed": hardtest, "attempts": hardtest})
            self.assertEqual(summary["answer_only"], {"passed": 0, "attempts": hardtest})
            self.assertEqual(summary["wrong_answer"], {"passed": 0, "attempts": hardtest})
            self.assertTrue(summary["gates_passed"])

    def test_scripted_check_accepts_solutions_and_rejects_negative_runs(self):
        with TemporaryDirectory() as temporary:
            out = Path(temporary) / "data"
            result = self.command("--output", out, "--check")
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((out / "summary.json").read_text())
            # Every train and test task (420 and 140 from Stage 8-9, plus the leveled Stage 10 train shapes).
            total = len(run.specs("train")) + len(run.specs("test"))
            self.assertGreaterEqual(total, 560)
            self.assertEqual(summary["correct"], {"passed": total, "attempts": total})
            self.assertEqual(summary["answer_only"], {"passed": 0, "attempts": total})
            self.assertEqual(summary["wrong_answer"], {"passed": 0, "attempts": total})
            self.assertTrue(summary["gates_passed"])

    def test_existing_output_is_never_overwritten(self):
        with TemporaryDirectory() as temporary:
            out = Path(temporary)
            sentinel = out / "keep.txt"
            sentinel.write_text("keep")
            result = self.command("--output", out)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_text(), "keep")

    def test_against_a_freeze_with_other_test_tasks_stops_before_writing(self):
        with TemporaryDirectory() as temporary:
            earlier = build_manifest()
            changed = next(task for task in earlier["tasks"] if task["split"] == "test")
            changed["prompt_sha256"] = "0" * 64
            old = Path(temporary) / "old.json"
            old.write_text(json.dumps(earlier))
            out = Path(temporary) / "data"
            result = self.command("--split", "test", "--output", out, "--against", old)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn(changed["id"], result.stderr)
            self.assertFalse(out.exists())

    def test_against_a_freeze_with_the_same_test_tasks_generates_the_data(self):
        with TemporaryDirectory() as temporary:
            old = Path(temporary) / "old.json"
            old.write_text(json.dumps(build_manifest()))
            out = Path(temporary) / "data"
            result = self.command("--split", "test", "--output", out, "--against", old)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((out / "manifest.json").is_file())

    def test_test_split_runs_require_final_and_a_manifest(self):
        with TemporaryDirectory() as temporary, patch.object(run, "load_model") as model:
            manifest = Path(temporary) / "manifest.json"
            manifest.write_text(json.dumps(build_manifest()))
            base = ["bench", "--split", "test", "--output", str(Path(temporary) / "run")]
            for flags in ([], ["--final"], ["--manifest", str(manifest)],
                          ["--final", "--manifest", str(manifest), "--limits", '{"planning": true}']):
                with self.subTest(flags=flags), patch.object(sys, "argv", base + flags):
                    with self.assertRaises(SystemExit) as error:
                        run.main()
                    self.assertEqual(error.exception.code, 2)
            model.assert_not_called()

    def test_supplied_manifest_drift_blocks_test_before_loading_model(self):
        with TemporaryDirectory() as temporary, patch.object(run, "load_model") as model:
            manifest = Path(temporary) / "manifest.json"
            content = build_manifest()
            content["memory"] = "on"
            manifest.write_text(json.dumps(content))
            args = ["bench", "--split", "test", "--final", "--manifest", str(manifest),
                    "--output", str(Path(temporary) / "run")]
            with patch.object(sys, "argv", args), self.assertRaises(SystemExit):
                run.main()
            model.assert_not_called()

    def test_final_needs_a_manifest_on_every_split(self):
        # Without --manifest there is no freeze made for this session to compare against; the
        # committed Stage 8 manifest is a test baseline, not one.
        with TemporaryDirectory() as temporary, patch.object(run, "load_model") as model, \
                patch("agent_from_scratch.evals.bench.manifest.manifest_drift", return_value={}):
            args = ["bench", "--split", "val", "--final", "--output", str(Path(temporary) / "run")]
            with patch.object(sys, "argv", args), self.assertRaises(SystemExit) as error:
                run.main()
            self.assertEqual(error.exception.code, 2)
            model.assert_not_called()

    def test_train_and_versioned_test_reach_the_existing_runner(self):
        for split in ("train", "test"):
            with self.subTest(split=split), TemporaryDirectory() as temporary:
                root = Path(temporary)
                skeleton, ctx = run.specs(split)[0]
                fixture = root / "fixture"
                fixture.mkdir()
                task = skeleton.build(random.Random(f"bench:{ctx.name}:{ctx.seed}:{ctx.condition}"), fixture, ctx)
                model = _model(skeleton.solution(task))
                model.settings.return_value = {"backend": "vllm", "model_root": BENCH_MODEL["id"],
                                               "n_ctx": BENCH_N_CTX, **BENCH_DECODING}
                manifest = root / "manifest.json"
                # Stands for a freeze made from committed code, whatever state this checkout is in.
                with patch("agent_from_scratch.evals.bench.manifest._worktree_clean", return_value=True):
                    manifest.write_text(json.dumps(build_manifest()))
                output = root / "run"
                args = ["bench", "--split", split, "--tasks", ctx.id, "--backend", "vllm",
                        "--output", str(output)]
                if split == "test":
                    args += ["--final", "--manifest", str(manifest)]
                with patch.object(sys, "argv", args), patch.object(run, "load_model", return_value=model):
                    run.main()
                record = json.loads((output / ctx.id / "result.json").read_text())
                self.assertTrue(record["passed"], record["checks"])

    def test_invalid_manifest_is_a_cli_error_before_loading_model(self):
        with TemporaryDirectory() as temporary, patch.object(run, "load_model") as model:
            manifest = Path(temporary) / "manifest.json"
            args = ["bench", "--split", "test", "--final", "--manifest", str(manifest),
                    "--output", str(Path(temporary) / "run")]
            for content in ("{broken", "[]", "null", '{"tasks": {}}', '{"tasks": [null]}'):
                with self.subTest(content=content):
                    manifest.write_text(content)
                    with patch.object(sys, "argv", args), self.assertRaises(SystemExit) as error:
                        run.main()
                    self.assertEqual(error.exception.code, 2)
            model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
