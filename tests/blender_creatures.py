"""Multi-legged, winged, serpentine and marine rigs: export, contact bones,
Check Rig messages and a rest-pose round trip through Apply."""
import sys, json
from pathlib import Path
import numpy as np
import bpy
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "addon"), str(ROOT / "tests")]
import unimate_motion
from unimate_motion.rig import export_skeleton, export_ground, apply_result, contact_names, detect_contact_bones, select_bones
from unimate_motion.motion import canonicalize, decode_features
from build_scene import build, build_creatures, build_objects

unimate_motion.register()
human, creature = build()
rigs = build_creatures()
objects = build_objects()
settings = bpy.context.scene.unimate_motion
out = ROOT / "tests" / "artifacts"
out.mkdir(parents=True, exist_ok=True)

def contacts(rig, tips=True):
    return contact_names(export_skeleton(rig, "-Y", tips))

def operate(rig, action):
    settings.rig = rig
    assert bpy.ops.unimate.contact(action=action) == {"FINISHED"}, settings.status

def check(rig, family="truebones", cleanup=True):
    settings.rig, settings.family, settings.motion_cleanup = rig, family, cleanup
    settings.forward, settings.tips, settings.fingers = "-Y", True, True
    try:
        bpy.ops.unimate.validate()
    except RuntimeError:
        pass
    return settings.status

# Rigs export within the model's joint limit, with flat wings and fins fitted by their thin side.
counts = {}
for name, rig in rigs.items():
    skeleton = export_skeleton(rig, "-Y", True)
    assert 5 <= len(skeleton["parents"]) <= 71, (name, len(skeleton["parents"]))
    counts[name] = len(skeleton["parents"])
dragon = export_skeleton(rigs["dragon"], "-Y", True)
wing = next(c for c in dragon["collision_capsules"] if dragon["bone_names"][c["joint"]] == "wing_fore.left")
assert wing["radius"] < .06, wing["radius"]  # 0.02 thick, 0.18 wide: the thin side

# Before detection only named feet count; steep limb tips get wide tilt limits once marked.
assert {n: len(contacts(r)) for n, r in rigs.items()} == dict(spider=0, crab=0, bird=2, dragon=0, snake=0, fish=0)
assert len(contacts(human)) == 2 and len(contacts(creature)) == 4
profile = export_skeleton(human, "-Y", True)["foot_profiles"][0]
assert abs(profile["stance_tilt"] - 29) < 1, profile  # unchanged for ordinary feet

# Detection finds leg tips, not tails, claws held up, wings, fins or the body of a snake.
expected = dict(spider=("tarsus", 8), crab=("dactyl", 6), bird=("foot", 2), dragon=("claw", 4), snake=("", 0), fish=("", 0))
for name, rig in rigs.items():
    found = detect_contact_bones(rig, "-Y")
    word, count = expected[name]
    new = [n for n in found if word not in n]
    assert not new, (name, found)
    if name != "bird":  # the bird's feet are already found by name
        assert len(found) == count, (name, found)
    else:
        assert found == [], found
    operate(rig, "DETECT")
    assert len(contacts(rig)) == count, (name, contacts(rig))
for rig in (human, creature):
    assert detect_contact_bones(rig, "-Y") == []
spider = export_skeleton(rigs["spider"], "-Y", True)
assert all(60 <= p["stance_tilt"] <= 120 for p in spider["foot_profiles"]), [p["stance_tilt"] for p in spider["foot_profiles"]]  # tips angled 47 degrees down; ordinary feet cap at 45
assert all(p["swing_tilt"] <= 150 for p in spider["foot_profiles"])
ground = export_ground(rigs["spider"], spider)
assert abs(ground["height"] - (0.03 - .015)) < .03, ground["height"]  # the tarsus tips just touch the ground

# Manual marks: a selected bone is added, an unmarked foot is dropped, Clear returns to names.
bird = rigs["bird"]
select_bones(bird, {"wing_hand.left"})
operate(bird, "MARK")
assert len(contacts(bird)) == 3
select_bones(bird, {"foot.left"})
operate(bird, "UNMARK")
assert sorted(contacts(bird)) == ["foot.right", "wing_hand.left"], contacts(bird)
operate(bird, "CLEAR")
assert sorted(contacts(bird)) == ["foot.left", "foot.right"]
select_bones(bird, set())
settings.rig = bird
try:
    bpy.ops.unimate.contact(action="MARK")  # nothing selected
    raise AssertionError("Mark with no selection accepted")
except RuntimeError as exc:
    assert "Select bones" in str(exc), exc

# Robots and plants: a quadruped robot's legs need detecting, an arm, a tracked robot and a plant have none.
assert {n: len(contacts(r)) for n, r in objects.items()} == dict(robot_arm=0, quad_robot=0, wheeled_robot=0, plant=0)
for name, rig in objects.items():
    skeleton = export_skeleton(rig, "-Y", True)
    assert 5 <= len(skeleton["parents"]) <= 71, (name, len(skeleton["parents"]))
    counts[name] = len(skeleton["parents"])
    found = detect_contact_bones(rig, "-Y")
    assert (sorted(found) == ["tip_fl", "tip_fr", "tip_hl", "tip_hr"]) == (name == "quad_robot"), (name, found)
    operate(rig, "DETECT")
assert len(contacts(objects["quad_robot"])) == 4
message = check(objects["robot_arm"], family="objaverse")
assert "0 contact bones" in message and "ground contact cleanup is skipped" in message, message
# Rigs with no contact bones in the "Other articulated model" family get a Fixed base hint, once it is on the hint goes.
assert "turn on Fixed base" in message, message
settings.fixed_base = True
assert "Fixed base" not in check(objects["robot_arm"], family="objaverse")
settings.fixed_base = False
assert "Fixed base" not in check(rigs["snake"]) and "Fixed base" not in check(objects["quad_robot"], family="objaverse")
assert "4 contact bones" in check(objects["quad_robot"], family="objaverse")
lines = unimate_motion.status_lines("Rig ready: 17 bones, 22/71 joints, general model, 2 contact bones")
assert all(len(l) <= 40 for l in lines) and " ".join(lines).endswith("general model, 2 contact bones"), lines
assert "not a biped" not in check(objects["quad_robot"], family="objaverse")

# Check Rig says how many contact bones it found and warns when cleanup would be skipped.
message = check(rigs["spider"])
assert "8 contact bones" in message and "No contact" not in message, message
settings.rig = rigs["snake"]
message = check(rigs["snake"])
assert "0 contact bones" in message and "ground contact cleanup is skipped" in message, message
assert "skipped" not in check(rigs["snake"], cleanup=False)
message = check(rigs["spider"], family="mixamo")
assert "not a biped" in message, message
assert "not a biped" not in check(human, family="mixamo")

# A rest pose decodes and applies onto every rig, whatever its bone rolls.
for name, rig in list(rigs.items()) + list(objects.items()) + [("human", human), ("creature", creature)]:
    skeleton = export_skeleton(rig, "-Y", True)
    canon = canonicalize(skeleton)
    features = np.zeros((3, len(skeleton["parents"]), 12))
    features[:, :, 3:9] = [1, 0, 0, 0, 1, 0]
    features[:, 0, 1] = canon["positions"][0, 1]
    positions, rotations = decode_features(features, canon)
    assert np.allclose(positions[0], skeleton["heads"], atol=1e-6), name
    path = out / f"creature-rest-{name}.npz"
    np.savez(path, schema=1, positions=positions, rotations=rotations, signature=skeleton["signature"],
             joint_names=np.array(skeleton["joint_names"]), fps=30., prompt="rest pose fixture", seed=0)
    apply_result(rig, skeleton, path, 1, bpy.context.scene)
    bpy.context.scene.frame_set(2)
    bpy.context.view_layer.update()
    for j, bone in enumerate(skeleton["bone_names"]):
        if bone:
            head = np.array(rig.matrix_world @ rig.pose.bones[bone].head) - np.array(rig.location)
            assert np.allclose(head, positions[1, j], atol=1e-4), (name, bone)

report = dict(passed=["joint counts", "thin-side capsules for wings", "contact bones by name", "contact detection",
                      "steep tip tilt limits", "mark, unmark and clear", "Check Rig contact messages", "Check Rig fixed base hint", "status wraps at words",
                      "robots and plants", "rest pose round trip"], joints=counts)
(out / "creatures-blender.json").write_text(json.dumps(report, indent=2))
print("CREATURES_BLENDER_PASSED", json.dumps(report))
unimate_motion.unregister()
