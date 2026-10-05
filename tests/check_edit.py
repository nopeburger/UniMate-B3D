"""Regenerating selected bones: whole-motion encoding and the pin masks (no Blender, no model)."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from timeline import edit_mask, edit_window, edit_windows, forward_kinematics, restore_kept, to_local
spec = importlib.util.spec_from_file_location("motion", ROOT / "addon/unimate_motion/motion.py")
motion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motion)
spec = importlib.util.spec_from_file_location("test_posecode", ROOT / "tests/test_posecode.py")
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)

skeleton = fixture.skeleton_with_clavicles()
parents = skeleton["parents"]
count, frames = len(parents), 40

# A motion with a curved, turning root path and moving limbs.
rng = np.random.default_rng(5)
t = np.arange(frames)
local = np.tile(np.eye(3), (frames, count, 1, 1))
local[:, 1:] = Rotation.from_rotvec(rng.normal(0, .25, (frames * (count - 1), 3))).as_matrix().reshape(frames, count - 1, 3, 3)
yaw = Rotation.from_euler("z", .04 * t).as_matrix()
local[:, 0] = yaw
root = np.stack([.02 * t * np.cos(.04 * t), .02 * t * np.sin(.04 * t), 1 + .01 * np.sin(t / 3)], axis=1) + np.asarray(skeleton["heads"][0])
positions, rotations = forward_kinematics(root, local, skeleton)

# decode_features(encode_motion(m)) must give m back.
features = motion.encode_motion(positions, rotations, skeleton)
assert features.shape == (frames, count, 12) and np.isfinite(features).all()
decoded_p, decoded_r = motion.decode_features(features, motion.canonicalize(skeleton))
# decode starts the root at the origin; the path is what matters.
decoded_p = decoded_p + (positions[0, 0] - decoded_p[0, 0]) * np.array([1, 1, 0])
# A leaf bone's rotation is stored nowhere (child slots hold their parent's rotation), so compare the others.
inner = [j for j in range(count) if j in parents]
assert np.allclose(decoded_r[:, inner], rotations[:, inner], atol=1e-6), np.abs(decoded_r[:, inner] - rotations[:, inner]).max()
assert np.allclose(decoded_p, positions, atol=1e-6), np.abs(decoded_p - positions).max()

# Masks: regenerating the left arm (and everything below it) pins the rest.
names = skeleton["bone_names"]
arm = names.index("mixamorig:LeftArm")
keep, regenerated = edit_mask(parents, [arm])
assert [names[j] for j in regenerated] == ["mixamorig:LeftArm", "mixamorig:LeftForeArm", "mixamorig:LeftHand"]
hand, shoulder = names.index("mixamorig:LeftHand"), names.index("mixamorig:LeftShoulder")
assert not keep[arm, 0:3].any() and not keep[hand, 0:3].any()          # their positions are regenerated
assert keep[shoulder, 0:3].all() and keep[names.index("mixamorig:RightArm"), 0:3].all()
# Slot j holds its parent's rotation: the arm's own rotation sits in the forearm slot and is regenerated,
# while the shoulder's rotation, held in the arm slot, stays pinned.
assert not keep[names.index("mixamorig:LeftForeArm"), 3:9].any()
assert keep[arm, 3:9].all() and keep[shoulder, 3:9].all()
assert keep[0].all()                                                    # root slot: facing, height, velocity
assert keep[:, 0:3][[j for j in range(count) if j not in regenerated]].all()
for bad in ([0], [99], list(range(1, count))):
    try:
        edit_mask(parents, bad); raise AssertionError("accepted " + str(bad))
    except ValueError:
        pass

# Windows: 40 frames in one 60-frame window; 130 frames need three (60, then 50 new each).
assert edit_windows(40, 60, 10) == 1 and edit_windows(60, 60, 10) == 1 and edit_windows(61, 60, 10) == 2
assert edit_windows(110, 60, 10) == 2 and edit_windows(111, 60, 10) == 3
normalized = rng.normal(size=(130, count, 12)).astype(np.float32)
known, mask = edit_window(normalized, keep, start=50, window=60, first_slot=10)
assert known.shape == mask.shape == (count, 12, 60)
assert not mask[:, :, :10].any()                                         # the previous window owns the overlap
assert np.array_equal(mask[:, :, 10], keep) and np.array_equal(known[:, :, 10], normalized[60])
assert np.allclose(known[:, :, 59], normalized[109])
known, mask = edit_window(normalized, keep, start=100, window=60, first_slot=10)
assert np.allclose(known[:, :, 59], normalized[129])                     # past the end repeats the last frame

# Kept bones and the root come back exactly as authored; regenerated bones keep their new motion.
other = Rotation.from_rotvec(rng.normal(0, .3, (frames * count, 3))).as_matrix().reshape(frames, count, 3, 3)
other_p, other_r = forward_kinematics(root + 1.0, other, skeleton)
fixed_p, fixed_r = restore_kept(other_p, other_r, positions, rotations, regenerated, skeleton)
mixed, authored, generated = to_local(fixed_r, parents), to_local(rotations, parents), to_local(other_r, parents)
for joint in range(count):
    source = generated if joint in regenerated else authored
    assert np.allclose(mixed[:, joint], source[:, joint], atol=1e-9), joint
assert np.allclose(fixed_p[:, 0], positions[:, 0])

report = dict(passed=["encode_motion round trip", "pin mask by slot ownership", "descendants regenerate with their parent",
                      "invalid selections rejected", "window counts", "window known values", "kept bones restored exactly"])
(ROOT / "tests" / "artifacts").mkdir(exist_ok=True)
(ROOT / "tests" / "artifacts" / "edit-regressions.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
