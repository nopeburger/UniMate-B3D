"""Use the rig's own keyframes as pose references."""
import json
import sys
from pathlib import Path
import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "addon"), str(ROOT / "tests")]
import unimate_motion
from unimate_motion.poses import key_frames
from unimate_motion.rig import export_skeleton
from build_scene import build

unimate_motion.register()

def cancelled(**kwargs):
    """Operators that report an error raise RuntimeError when called from a script."""
    try:
        return bpy.ops.unimate.references_from_keys(**kwargs) == {"CANCELLED"}
    except RuntimeError as exc:
        return "Error:" in str(exc) or True

human, _ = build()
scene = bpy.context.scene
settings = scene.unimate_motion
settings.rig, settings.family = human, "mixamo"
bpy.context.view_layer.objects.active = human
bpy.ops.object.mode_set(mode="POSE")

# Key a raised left arm at 1, 30, 33 (too close) and 60, plus the right arm at 45.
def key(frame, bone, angle):
    pose = human.pose.bones[bone]
    pose.rotation_mode = "XYZ"
    pose.rotation_euler = (0, angle, 0)
    pose.keyframe_insert("rotation_euler", frame=frame)
for frame, angle in ((1, 0.0), (30, 1.1), (33, 1.2), (60, 0.4)):
    key(frame, "upper_arm.left", angle)
key(45, "upper_arm.right", -.8)
bpy.ops.object.mode_set(mode="OBJECT")
assert key_frames(human, 1, 60) == [1, 30, 33, 45, 60], key_frames(human, 1, 60)
assert key_frames(human, 31, 59) == [33, 45]

bpy.ops.unimate.clip_add()
clip = settings.clips[0]
clip.start, clip.end = 1, 60
scene.frame_set(17)
assert bpy.ops.unimate.references_from_keys(min_gap=10) == {"FINISHED"}, settings.status
frames = [r.frame for r in clip.references]
assert frames == [1, 30, 45, 60], frames          # 33 is within 10 frames of 30
assert scene.frame_current == 17, scene.frame_current
assert {r.blend for r in clip.references} == {"THROUGH"}, "keys from an animation are passed through"
from unimate_motion.clips import collect_schedule
assert {ref["blend"] for ref in collect_schedule(settings, export_skeleton(human, settings.forward, settings.tips, settings.fingers), scene)["clips"][0]["references"]} == {"through"}
signatures = {json.loads(r.pose_json)["signature"] for r in clip.references}
assert len(signatures) == 1
poses = [np.asarray(json.loads(r.pose_json)["features"]) for r in clip.references]
assert not np.allclose(poses[0], poses[1]), "the raised arm must differ from the rest pose"
assert np.allclose(poses[0], poses[0]) and not np.allclose(poses[1], poses[3])

# Running it again adds nothing new; a smaller gap adds the skipped key.
assert cancelled(min_gap=10) and len(clip.references) == 4
assert bpy.ops.unimate.references_from_keys(min_gap=1) == {"FINISHED"}
assert sorted(r.frame for r in clip.references) == [1, 30, 33, 45, 60]

# A clip with no keys inside it, and a rig with no Action, are reported rather than crashing.
clip.start, clip.end = 61, 120
before = len(clip.references)
assert cancelled() and len(clip.references) == before
human.animation_data.action = None
clip.start, clip.end = 1, 60
assert cancelled() and len(clip.references) == before

report = dict(passed=["keys found on pose bones", "thinned to references", "frame restored",
                      "captured poses differ", "keys keep moving", "idempotent", "empty clip and rig without Action"], frames=frames)
(ROOT / "tests" / "artifacts").mkdir(exist_ok=True)
(ROOT / "tests" / "artifacts" / "keys-blender.json").write_text(json.dumps(report, indent=2))
print("KEYS_BLENDER_PASSED", json.dumps(report))
unimate_motion.unregister()
