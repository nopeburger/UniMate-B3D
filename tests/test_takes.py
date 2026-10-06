"""Several takes per Generate: the worker's take loop, with a stand-in for the model."""
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("worker", ROOT / "backend" / "worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class TakesTest(unittest.TestCase):
    def run_takes(self, takes, fail_on=None):
        folder = Path(tempfile.mkdtemp())
        request = folder / "request.json"
        request.write_text(json.dumps(dict(seed=10, takes=takes)), encoding="utf-8")
        calls, statuses = [], []
        def fake_generate(req, output, status):
            worker.write_status(status, "running", "Generating window 1/1")
            statuses.append(json.loads(status.read_text())["message"])
            if req["seed"] == fail_on:
                raise RuntimeError("model failed")
            output.write_text(str(req["seed"]))
            calls.append((req["seed"], output.name))
            return "Motion ready — Apply Motion to create an Action" + (" (review: 3 collision frames)" if req["seed"] == 11 else "")
        original, worker.generate = worker.generate, fake_generate
        try:
            # The worker prints its status and, for failures, a traceback; keep the test output clean.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = worker.run_job(request, folder / "motion.npz", folder / "status.json")
        finally:
            worker.generate = original
        return code, calls, statuses, json.loads((folder / "status.json").read_text())

    def test_one_take_is_unchanged(self):
        code, calls, statuses, final = self.run_takes(1)
        self.assertEqual((code, calls), (0, [(10, "motion.npz")]))
        self.assertEqual(statuses, ["Generating window 1/1"])
        self.assertEqual(final["state"], "complete")
        self.assertTrue(final["message"].startswith("Motion ready"))

    def test_takes_use_consecutive_seeds_and_files(self):
        code, calls, statuses, final = self.run_takes(3)
        self.assertEqual(code, 0)
        self.assertEqual(calls, [(10, "motion.npz"), (11, "motion_take2.npz"), (12, "motion_take3.npz")])
        self.assertEqual(statuses, [f"Take {k}/3: Generating window 1/1" for k in (1, 2, 3)])
        self.assertEqual(final["state"], "complete")
        self.assertIn("3 takes ready (seeds 10-12)", final["message"])
        self.assertIn("1 need review", final["message"])

    def test_a_failing_take_fails_the_job(self):
        code, calls, statuses, final = self.run_takes(3, fail_on=11)
        self.assertEqual(code, 1)
        self.assertEqual(final["state"], "failed")
        self.assertNotIn("Take", final["message"])

    def test_take_count_is_limited(self):
        for takes in (0, 9):
            code, calls, statuses, final = self.run_takes(takes)
            self.assertEqual((code, calls, final["state"]), (1, [], "failed"))


if __name__ == "__main__":
    unittest.main()
