"""Regenerate Selected Bones: capturing the existing motion and building the request."""
import json
import sys
from pathlib import Path
import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "addon"), str(ROOT / "tests")]
import unimate_motion
from unimate_motion import clips
from unimate_motion.motion import decode_features, canonicalize
from unimate_motion.poses import capture_motion, capture_pose
from unimate_motion.rig import export_skeleton, select_bones
from build_scene import build

unimate_motion.register()
human, _ = build()
scene = bpy.context.scene
settings = scene.unimate_motion
settings.rig, settings.family = human, "mixamo"
bpy.context.view_layer.objects.active = human
bpy.ops.object.mode_set(mode="POSE")
for frame, (left, right, hips) in ((1, (0, 0, 0)), (20, (1.0, -.5, .1)), (40, (.2, -1.2, -.1))):
    for bone, angle in (("upper_arm.left", left), ("upper_arm.right", right), ("thigh.left", hips)):
        pose = human.pose.bones[bone]
        pose.rotation_mode = "XYZ"
        pose.rotation_euler = (0, angle, 0)
        pose.keyframe_insert("rotation_euler", frame=frame)
human.pose.bones["hips"].location = (0, 0, 0)
human.pose.bones["hips"].keyframe_insert("location", frame=1)
human.pose.bones["hips"].location = (.4, .2, 0)
human.pose.bones["hips"].keyframe_insert("location", frame=40)
bpy.ops.object.mode_set(mode="OBJECT")

skeleton = export_skeleton(human, settings.forward, settings.tips, settings.fingers)
scene.frame_set(7)
positions, rotations = capture_motion(human, skeleton, scene, 1, 40)
assert positions.shape == (40, len(skeleton["parents"]), 3) and rotations.shape[-2:] == (3, 3)
assert scene.frame_current == 7
scene.frame_set(20)
assert np.allclose(capture_pose(human, skeleton)["features"],
                   __import__("unimate_motion.motion", fromlist=["x"]).encode_pose(positions[19], rotations[19], skeleton)["features"])
# The root travelled, and the whole thing decodes back to the same path.
assert np.linalg.norm(positions[-1, 0] - positions[0, 0]) > .3
from unimate_motion.motion import encode_motion
features = encode_motion(positions, rotations, skeleton)
decoded, _ = decode_features(features, canonicalize(skeleton))
decoded = decoded + (positions[0, 0] - decoded[0, 0])
assert np.allclose(decoded, positions, atol=1e-5), np.abs(decoded - positions).max()

def schedule(**changes):
    clip = dict(prompt="A person waves.", start=1, end=40, references=[])
    clip.update(changes)
    return [clip]

def fails(message, **kwargs):
    try:
        clips.collect_edit(settings, human, skeleton, scene, **kwargs)
    except ValueError as exc:
        assert message in str(exc), str(exc)
        return
    raise AssertionError("accepted: " + message)

bpy.ops.object.mode_set(mode="POSE")
select_bones(human, [])
fails("Select the bones", clip_list=schedule())
select_bones(human, ["hips"])
fails("root bone cannot be regenerated", clip_list=schedule())
select_bones(human, ["upper_arm.right"])
fails("one prompt clip", clip_list=schedule() + schedule(start=41, end=80))
fails("Remove the pose references", clip_list=schedule(references=[dict(frame=10, pose={})]))
edit = clips.collect_edit(settings, human, skeleton, scene, schedule())
assert edit["regenerate"] == ["upper_arm.right"]
assert np.asarray(edit["features"]).shape == (40, len(skeleton["parents"]), 12)
assert np.allclose(edit["root_start"], positions[0, 0])
human.animation_data.action = None
fails("no Action", clip_list=schedule())

report = dict(passed=["capture_motion matches capture_pose", "scene frame restored", "motion round trip",
                      "selection and clip validation", "request payload"])
(ROOT / "tests" / "artifacts").mkdir(exist_ok=True)
(ROOT / "tests" / "artifacts" / "edit-blender.json").write_text(json.dumps(report, indent=2))
print("EDIT_BLENDER_PASSED", json.dumps(report))
unimate_motion.unregister()
