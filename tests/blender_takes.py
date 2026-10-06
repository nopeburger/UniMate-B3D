"""Several takes per Generate in Blender: Apply makes one Action per take, Take switches between them."""
import json
import shutil
import sys
import tempfile
from pathlib import Path
import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "addon"), str(ROOT / "tests")]
import unimate_motion
from build_scene import build

unimate_motion.register()
human, _ = build()
scene = bpy.context.scene
settings = scene.unimate_motion
fixture = ROOT / "tests" / "artifacts" / "seated-transition"   # written by make_transition_fixture.py
request = json.loads((fixture / "request.json").read_text(encoding="utf-8"))
scene.render.fps, scene.render.fps_base = request["fps"], 1

job = Path(tempfile.mkdtemp())
shutil.copy(fixture / "request.json", job / "request.json")
for name in ("motion.npz", "motion_take2.npz", "motion_take3.npz"):
    shutil.copy(fixture / "motion.npz", job / name)
settings.rig, settings.job_dir = human, str(job)
bpy.context.view_layer.objects.active = human
assert settings.takes == 1 and settings.applied_takes == 0
assert unimate_motion.take_files(job) == [job / "motion.npz", job / "motion_take2.npz", job / "motion_take3.npz"]

before = len(bpy.data.actions)
assert bpy.ops.unimate.apply() == {"FINISHED"}, settings.status
takes = [a for a in bpy.data.actions if a.get("unimate_job") == str(job)]
assert len(bpy.data.actions) - before == 3 and sorted(a["unimate_take"] for a in takes) == [1, 2, 3]
assert all(a.name.endswith(f"| take {a['unimate_take']}") for a in takes)
assert human.animation_data.action["unimate_take"] == 1, "take 1 plays after Apply"
assert settings.applied_takes == 3 and settings.take == 1
assert "3 Actions" in settings.status

settings.take = 3
assert human.animation_data.action["unimate_take"] == 3
scene.frame_set(40)                                           # the take really plays (it is evaluated)
settings.take = 2
assert human.animation_data.action["unimate_take"] == 2
settings.take = 5
assert human.animation_data.action["unimate_take"] == 2 and "not been applied" in settings.status

# A single-take job still makes exactly one Action, named as before.
single = Path(tempfile.mkdtemp())
shutil.copy(fixture / "request.json", single / "request.json")
shutil.copy(fixture / "motion.npz", single / "motion.npz")
settings.job_dir = str(single)
assert bpy.ops.unimate.apply() == {"FINISHED"}
assert settings.applied_takes == 1 and "| take" not in human.animation_data.action.name

report = dict(passed=["one Action per take", "take 1 plays after Apply", "Take switches the playing Action",
                      "missing take reported", "single take unchanged"])
(ROOT / "tests" / "artifacts" / "takes-blender.json").write_text(json.dumps(report, indent=2))
print("TAKES_BLENDER_PASSED", json.dumps(report))
unimate_motion.unregister()
