import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]
PACKAGE = "unimate_motion_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "addon" / "unimate_motion")]
sys.modules.setdefault(PACKAGE, package)


def load(name):
    spec = importlib.util.spec_from_file_location(
        f"{PACKAGE}.{name}", ROOT / "addon" / "unimate_motion" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


motion = load("motion")
posecode = load("posecode")


NAMES = [
    "mixamorigHips", "mixamorig:Spine", "mixamorig:Spine2", "mixamorig:Neck", "mixamorig:Head",
    "mixamorig:LeftArm", "mixamorig:LeftForeArm", "mixamorig:LeftHand",
    "mixamorig:RightArm", "mixamorig:RightForeArm", "mixamorig:RightHand",
    "mixamorig:LeftUpLeg", "mixamorig:LeftLeg", "mixamorig:LeftFoot",
    "mixamorig:RightUpLeg", "mixamorig:RightLeg", "mixamorig:RightFoot",
]
PARENTS = [-1, 0, 1, 2, 3, 2, 5, 6, 2, 8, 9, 0, 11, 12, 0, 14, 15]
HEADS = [
    (0, 0, 1), (0, 0, 1.2), (0, 0, 1.4), (0, 0, 1.6), (0, 0, 1.75),
    (.2, 0, 1.5), (.45, 0, 1.5), (.7, 0, 1.5),
    (-.2, 0, 1.5), (-.45, 0, 1.5), (-.7, 0, 1.5),
    (.1, 0, .95), (.1, 0, .5), (.1, 0, .08),
    (-.1, 0, .95), (-.1, 0, .5), (-.1, 0, .08),
]


def skeleton():
    matrices = []
    for head in HEADS:
        matrix = np.eye(4)
        matrix[:3, 3] = head
        matrices.append(matrix.tolist())
    return {
        "forward": "-Y", "joint_names": NAMES, "bone_names": NAMES,
        "parents": PARENTS, "heads": HEADS, "rest_matrices": matrices,
        "signature": "fixture-signature",
    }


def skeleton_with_clavicles():
    names = [
        "mixamorigHips", "mixamorig:Spine", "mixamorig:Spine2", "mixamorig:Neck", "mixamorig:Head",
        "mixamorig:LeftShoulder", "mixamorig:LeftArm", "mixamorig:LeftForeArm", "mixamorig:LeftHand",
        "mixamorig:RightShoulder", "mixamorig:RightArm", "mixamorig:RightForeArm", "mixamorig:RightHand",
        "mixamorig:LeftUpLeg", "mixamorig:LeftLeg", "mixamorig:LeftFoot",
        "mixamorig:RightUpLeg", "mixamorig:RightLeg", "mixamorig:RightFoot",
    ]
    parents = [-1, 0, 1, 2, 3, 2, 5, 6, 7, 2, 9, 10, 11, 0, 13, 14, 0, 16, 17]
    heads = [
        (0, 0, 1), (0, 0, 1.2), (0, 0, 1.4), (0, 0, 1.6), (0, 0, 1.75),
        (.1, 0, 1.5), (.2, 0, 1.5), (.45, 0, 1.5), (.7, 0, 1.5),
        (-.1, 0, 1.5), (-.2, 0, 1.5), (-.45, 0, 1.5), (-.7, 0, 1.5),
        (.1, 0, .95), (.1, 0, .5), (.1, 0, .08),
        (-.1, 0, .95), (-.1, 0, .5), (-.1, 0, .08),
    ]
    matrices = []
    for head in heads:
        matrix = np.eye(4)
        matrix[:3, 3] = head
        matrices.append(matrix.tolist())
    return {
        "forward": "-Y", "joint_names": names, "bone_names": names,
        "parents": parents, "heads": heads, "rest_matrices": matrices,
        "signature": "clavicle-fixture-signature",
    }


def manifest():
    bindings = [
        ("pelvis", "Hips"), ("spine", "Spine"), ("chest", "Spine2"),
        ("neck", "Neck"), ("head", "Head"),
        ("shoulder_left", "LeftArm"), ("elbow_left", "LeftForeArm"), ("wrist_left", "LeftHand"),
        ("shoulder_right", "RightArm"), ("elbow_right", "RightForeArm"), ("wrist_right", "RightHand"),
        ("hip_left", "LeftUpLeg"), ("knee_left", "LeftLeg"), ("ankle_left", "LeftFoot"),
        ("hip_right", "RightUpLeg"), ("knee_right", "RightLeg"), ("ankle_right", "RightFoot"),
    ]
    empty = {"groundLock": [], "reaches": [], "pins": [], "grips": []}
    return {
        "schema": posecode.SCHEMA,
        "timing": {"fps": 30, "frames": 90, "durationSeconds": 3, "frameIndexing": "one-based-inclusive"},
        "rig": {"profile": "posecode-humanoid", "boneBindings": [
            {"posecode": source, "mixamo": target} for source, target in bindings
        ]},
        "clips": [
            {"prompt": "lower into a squat", "start": 1, "end": 30, "constraints": empty,
             "references": [{"frame": 1, "keyframeId": "start"}, {"frame": 30, "keyframeId": "lower"}]},
            {"prompt": "reach to the left", "start": 31, "end": 60, "constraints": empty,
             "references": [{"frame": 60, "keyframeId": "reach"}]},
            {"prompt": "stand and raise the left arm", "start": 61, "end": 90, "constraints": empty,
             "references": [{"frame": 90, "keyframeId": "finish"}]},
        ],
        "keyframes": [
            {"id": "start", "frame": 1, "localEulerDeg": {},
             "root": {"positionMeters": [0, 0, 0], "rotationDeg": [0, 0, 0], "yawDeg": 0},
             "constraints": empty},
            {"id": "lower", "frame": 30,
             "localEulerDeg": {"hip_left": [-45, 0, 0], "hip_right": [-45, 0, 0],
                               "knee_left": [70, 0, 0], "knee_right": [70, 0, 0]},
             "root": {"positionMeters": [0, -.17, 0], "rotationDeg": [0, 0, 0], "yawDeg": 0},
             "constraints": empty},
            {"id": "reach", "frame": 60,
             "localEulerDeg": {"shoulder_left": [-45, 0, 30], "elbow_left": [-30, 0, 0]},
             "root": {"positionMeters": [0, 0, 0], "rotationDeg": [0, 0, 0], "yawDeg": 15},
             "constraints": empty},
            {"id": "finish", "frame": 90, "localEulerDeg": {"shoulder_left": [0, 0, 90]},
             "root": {"positionMeters": [0, 0, 0], "rotationDeg": [0, 0, 0], "yawDeg": 0},
             "constraints": empty},
        ],
    }


class PosecodeBridgeTests(unittest.TestCase):
    def test_builds_captured_references_for_selected_rig(self):
        schedule = posecode.build_schedule(manifest(), skeleton())
        self.assertEqual([(clip["start"], clip["end"]) for clip in schedule],
                         [(1, 30), (31, 60), (61, 90)])
        references = [reference for clip in schedule for reference in clip["references"]]
        self.assertEqual([ref["frame"] for ref in references], [1, 30, 60, 90])
        for reference in references:
            pose = reference["pose"]
            self.assertEqual(pose["signature"], "fixture-signature")
            self.assertEqual(np.asarray(pose["features"]).shape, (17, 12))
            self.assertTrue(np.isfinite(pose["features"]).all())
        start = np.asarray(references[0]["pose"]["features"])
        end = np.asarray(references[-1]["pose"]["features"])
        self.assertFalse(np.allclose(start, end))
        positions, _ = motion.decode_features(end[None], motion.canonicalize(skeleton()))
        # The +90 degree semantic shoulder Z rotation raises the left forearm
        # in the selected rig instead of applying Mixamo's raw local axes.
        self.assertGreater(positions[0, 6, 2], positions[0, 5, 2])

        lower = np.asarray(references[1]["pose"]["features"])
        lower_positions, _ = motion.decode_features(lower[None], motion.canonicalize(skeleton()))
        self.assertAlmostEqual(lower_positions[0, 13, 2], HEADS[13][2], delta=.005)
        self.assertAlmostEqual(lower_positions[0, 16, 2], HEADS[16][2], delta=.005)

    def test_shares_swing_with_clavicle_without_changing_upper_arm_world_rotation(self):
        data = manifest()
        keyframe = data["keyframes"][-1]
        base = skeleton()
        shared = skeleton_with_clavicles()

        base_pose = posecode.encode_keyframe(
            keyframe, base, posecode._match_bones(data["rig"]["boneBindings"], base)
        )
        shared_pose = posecode.encode_keyframe(
            keyframe, shared, posecode._match_bones(data["rig"]["boneBindings"], shared)
        )
        _, base_rotations = motion.decode_features(
            np.asarray(base_pose["features"])[None], motion.canonicalize(base)
        )
        _, shared_rotations = motion.decode_features(
            np.asarray(shared_pose["features"])[None], motion.canonicalize(shared)
        )

        base_arm = base["bone_names"].index("mixamorig:LeftArm")
        shared_arm = shared["bone_names"].index("mixamorig:LeftArm")
        clavicle = shared["bone_names"].index("mixamorig:LeftShoulder")
        chest = shared["bone_names"].index("mixamorig:Spine2")
        np.testing.assert_allclose(
            shared_rotations[0, shared_arm], base_rotations[0, base_arm], atol=1e-7
        )
        clavicle_delta = shared_rotations[0, chest].T @ shared_rotations[0, clavicle]
        clavicle_angle = math.acos(
            np.clip((np.trace(clavicle_delta) - 1) / 2, -1.0, 1.0)
        )
        self.assertGreater(clavicle_angle, math.radians(1))
        self.assertLessEqual(clavicle_angle, posecode.CLAVICLE_MAX_UP + 1e-7)

    def test_rejects_unknown_reference(self):
        data = manifest()
        data["clips"][0]["references"][0]["keyframeId"] = "missing"
        with self.assertRaisesRegex(ValueError, "unknown keyframe"):
            posecode.validate_manifest(data)

    def test_rejects_non_mixamo_rig(self):
        data = manifest()
        data["rig"]["boneBindings"] = [
            binding for binding in data["rig"]["boneBindings"] if binding["posecode"] != "head"
        ]
        with self.assertRaisesRegex(ValueError, "missing required Posecode bones"):
            posecode.build_schedule(data, skeleton())


if __name__ == "__main__":
    unittest.main()
