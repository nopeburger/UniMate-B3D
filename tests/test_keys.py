"""Thinning keyframes into pose reference frames (no Blender needed)."""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("schedule", Path(__file__).resolve().parents[1] / "addon/unimate_motion/schedule.py")
schedule = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schedule)
thin = schedule.thin_frames


class ThinFramesTest(unittest.TestCase):
    def test_keeps_widely_spaced_frames(self):
        self.assertEqual(thin([1, 30, 60], 10), [1, 30, 60])

    def test_drops_frames_too_close_to_the_previous_one(self):
        self.assertEqual(thin([1, 5, 12, 30, 60], 10), [1, 12, 30, 60])

    def test_dense_keys_are_sampled_and_the_last_key_is_kept(self):
        self.assertEqual(thin(list(range(1, 61)), 10), [1, 11, 21, 31, 41, 60])   # 51 gives way to the last key, 9 frames later

    def test_last_key_replaces_a_middle_key_that_is_too_close(self):
        self.assertEqual(thin([1, 25, 30], 10), [1, 30])

    def test_duplicates_unsorted_and_short_inputs(self):
        self.assertEqual(thin([30, 1, 30, 1], 10), [1, 30])
        self.assertEqual(thin([7], 10), [7])
        self.assertEqual(thin([], 10), [])
        self.assertEqual(thin([1, 3], 10), [1, 3])


if __name__ == "__main__":
    unittest.main()
