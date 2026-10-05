"""Import Posecode sparse constraints into UniMate's pose-reference schedule.

This module has no Blender dependency so the interchange and retargeting math
can be tested with ordinary Python.  Posecode rotations are expressed in its
Y-up, +Z-forward anatomical driver.  We compose those rotations there, move
the resulting world-space deltas into the selected rig's canonical frame, run
FK on that rig's rest offsets, then reuse UniMate-B3D's pose encoder.
"""
import json
import math
from pathlib import Path
import re

import numpy as np

from .motion import canonicalize, encode_pose, semantic_name


SCHEMA = "posecode.unimate.constraints.v1"

_PARENTS = {
    "pelvis": None,
    "spine": "pelvis",
    "chest": "spine",
    "neck": "chest",
    "head": "neck",
    "shoulder_left": "chest",
    "elbow_left": "shoulder_left",
    "wrist_left": "elbow_left",
    "shoulder_right": "chest",
    "elbow_right": "shoulder_right",
    "wrist_right": "elbow_right",
    "hip_left": "pelvis",
    "knee_left": "hip_left",
    "ankle_left": "knee_left",
    "hip_right": "pelvis",
    "knee_right": "hip_right",
    "ankle_right": "knee_right",
}
for _side in ("left", "right"):
    for _finger in ("thumb", "index", "middle", "ring", "pinky"):
        _PARENTS[f"{_finger}_{_side}"] = f"wrist_{_side}"


def load_manifest(path):
    """Read and validate a Posecode constraint-manifest JSON file."""
    path = Path(path)
    if path.stat().st_size > 5 * 1024 * 1024:
        raise ValueError("Posecode manifest exceeds the 5 MB import limit.")
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(data)
    return data


def validate_manifest(data):
    """Validate the bounded subset consumed by the Blender adapter."""
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError(f"Expected a {SCHEMA} manifest.")
    timing = data.get("timing")
    if not isinstance(timing, dict) or timing.get("frameIndexing") != "one-based-inclusive":
        raise ValueError("Posecode manifest must use one-based inclusive frames.")
    fps = timing.get("fps")
    frames = timing.get("frames")
    if not isinstance(fps, int) or not 1 <= fps <= 240:
        raise ValueError("Posecode manifest FPS must be an integer from 1 to 240.")
    if not isinstance(frames, int) or not 1 <= frames <= 10000:
        raise ValueError("Posecode manifest must contain 1 to 10,000 frames.")
    rig = data.get("rig")
    if not isinstance(rig, dict) or rig.get("profile") != "posecode-humanoid":
        raise ValueError("This importer currently supports the posecode-humanoid profile only.")
    bindings = rig.get("boneBindings")
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("Posecode manifest has no bone bindings.")
    seen_bindings = set()
    for binding in bindings:
        if not isinstance(binding, dict) or not all(
            isinstance(binding.get(key), str) and binding[key].strip()
            for key in ("posecode", "mixamo")
        ):
            raise ValueError("Every Posecode bone binding needs posecode and mixamo names.")
        if binding["posecode"] in seen_bindings:
            raise ValueError(f"Duplicate Posecode binding: {binding['posecode']}")
        seen_bindings.add(binding["posecode"])
    keyframes = data.get("keyframes")
    if not isinstance(keyframes, list) or not keyframes:
        raise ValueError("Posecode manifest has no keyframes.")
    ids = set()
    for keyframe in keyframes:
        if not isinstance(keyframe, dict) or not isinstance(keyframe.get("id"), str):
            raise ValueError("Every Posecode keyframe needs an id.")
        if keyframe["id"] in ids:
            raise ValueError(f"Duplicate Posecode keyframe id: {keyframe['id']}")
        ids.add(keyframe["id"])
        frame = keyframe.get("frame")
        if not isinstance(frame, int) or not 1 <= frame <= frames:
            raise ValueError(f"Keyframe {keyframe['id']} lies outside the timeline.")
        rotations = keyframe.get("localEulerDeg")
        if not isinstance(rotations, dict):
            raise ValueError(f"Keyframe {keyframe['id']} has no local rotations.")
        for bone, angles in rotations.items():
            if bone not in _PARENTS:
                raise ValueError(f"Unsupported Posecode bone: {bone}")
            _vector(angles, f"rotation for {bone}")
        root = keyframe.get("root")
        if not isinstance(root, dict):
            raise ValueError(f"Keyframe {keyframe['id']} has no root transform.")
        _vector(root.get("positionMeters"), "root position")
        _vector(root.get("rotationDeg"), "root rotation")
        if not _finite(root.get("yawDeg")):
            raise ValueError("Posecode root yaw must be finite.")
    clips = data.get("clips")
    if not isinstance(clips, list) or not 1 <= len(clips) <= 32:
        raise ValueError("Posecode manifest must contain 1 to 32 clips.")
    previous = None
    for clip in clips:
        if not isinstance(clip, dict) or not isinstance(clip.get("prompt"), str) or not clip["prompt"].strip():
            raise ValueError("Every Posecode clip needs a prompt.")
        start, end = clip.get("start"), clip.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or not 1 <= start < end <= frames:
            raise ValueError("Every Posecode clip needs a valid range of at least two frames.")
        if end - start + 1 > 600:
            raise ValueError("Posecode clips may contain at most 600 frames.")
        if previous is not None and start != previous + 1:
            raise ValueError("Posecode clip ranges must be contiguous and ordered.")
        used_frames = set()
        for reference in clip.get("references", []):
            if not isinstance(reference, dict) or reference.get("keyframeId") not in ids:
                raise ValueError("Posecode clip refers to an unknown keyframe.")
            frame = reference.get("frame")
            if not isinstance(frame, int) or not start <= frame <= end or frame in used_frames:
                raise ValueError("Posecode reference frames must be unique and inside their clip.")
            used_frames.add(frame)
        previous = end
    return data


def build_schedule(manifest, skeleton):
    """Return UniMate-B3D clips with encoded poses for one exported rig."""
    validate_manifest(manifest)
    mapping = _match_bones(manifest["rig"]["boneBindings"], skeleton)
    keyframes = {item["id"]: item for item in manifest["keyframes"]}
    poses = {
        keyframe_id: encode_keyframe(keyframe, skeleton, mapping)
        for keyframe_id, keyframe in keyframes.items()
    }
    clips = []
    for source in manifest["clips"]:
        references = [
            {
                "frame": reference["frame"],
                "image_path": "",
                "pose": poses[reference["keyframeId"]],
            }
            for reference in source.get("references", [])
        ]
        clips.append({
            "prompt": source["prompt"].strip(),
            "start": source["start"],
            "end": source["end"],
            "references": references,
        })
    return clips


def encode_keyframe(keyframe, skeleton, mapping):
    """Retarget one semantic keyframe and encode it as a B3D pose reference."""
    canonical = canonicalize(skeleton)
    basis = canonical["basis"]
    parents = canonical["parents"]
    heads = np.asarray(skeleton["heads"], dtype=float)
    rotations = keyframe["localEulerDeg"]
    root = keyframe["root"]

    base = _euler_xyz(root["rotationDeg"])
    yaw = _axis_y(math.radians(root["yawDeg"]))
    root_delta = yaw @ base
    semantic = {}

    def compose(name):
        if name in semantic:
            return semantic[name]
        local = _euler_xyz(rotations.get(name, (0, 0, 0)))
        parent = _PARENTS[name]
        semantic[name] = (root_delta if parent is None else compose(parent)) @ local
        return semantic[name]

    for name in _PARENTS:
        compose(name)

    by_index = {index: name for name, index in mapping.items()}
    deltas = np.zeros((len(parents), 3, 3), dtype=float)
    for index, parent in enumerate(parents):
        if index in by_index:
            canonical_delta = semantic[by_index[index]]
            deltas[index] = basis.T @ canonical_delta @ basis
        elif parent >= 0:
            deltas[index] = deltas[parent]
        else:
            deltas[index] = basis.T @ root_delta @ basis

    for side in ("left", "right"):
        _share_with_clavicle(deltas, parents, heads, mapping, by_index, side)

    positions = np.zeros_like(heads)
    translation = basis.T @ np.asarray(root["positionMeters"], dtype=float)
    positions[0] = heads[0] + translation
    for index, parent in enumerate(parents[1:], 1):
        positions[index] = positions[parent] + deltas[parent] @ (heads[index] - heads[parent])
    return encode_pose(positions, deltas, skeleton)


# Shoulder rhythm: the clavicle/scapula carry about a third of upper-arm
# elevation. Posecode has no clavicle, so without this the whole raise lands
# on the ball joint and skinned shoulders pinch.
CLAVICLE_SHARE = 1 / 3
CLAVICLE_MAX_UP = math.radians(30)
CLAVICLE_MAX_DOWN = math.radians(10)


def _share_with_clavicle(deltas, parents, heads, mapping, by_index, side):
    """Share upper-arm swing with an unbound clavicle without changing arm rotation."""
    arm = mapping.get(f"shoulder_{side}")
    if arm is None:
        return
    clavicle = parents[arm]
    if clavicle <= 0 or clavicle in by_index:
        return
    chest = parents[clavicle]
    relative = deltas[chest].T @ deltas[arm]
    axis, angle = _axis_angle(relative)
    if angle < 1e-6:
        return

    # Swing only: twist about the upper arm's rest direction stays at the arm.
    children = [joint for joint, parent in enumerate(parents) if parent == arm]
    if not children:
        return
    bone = heads[children[0]] - heads[arm]
    bone /= np.linalg.norm(bone)
    swing = axis * angle
    swing -= bone * (swing @ bone)
    angle = float(np.linalg.norm(swing))
    if angle < 1e-6:
        return
    raised = (_rotation(swing / angle, angle) @ bone)[2] > bone[2]
    limit = CLAVICLE_MAX_UP if raised else CLAVICLE_MAX_DOWN
    deltas[clavicle] = deltas[chest] @ _rotation(
        swing / angle, min(angle * CLAVICLE_SHARE, limit)
    )


def _axis_angle(matrix):
    angle = math.acos(max(-1.0, min(1.0, (np.trace(matrix) - 1) / 2)))
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0]), 0.0
    if math.pi - angle < 1e-4:
        # Near 180 degrees the antisymmetric part vanishes; use the diagonal.
        axis = np.sqrt(np.maximum((np.diag(matrix) + 1) / 2, 0))
        axis[1] = math.copysign(axis[1], matrix[0, 1] + matrix[1, 0])
        axis[2] = math.copysign(axis[2], matrix[0, 2] + matrix[2, 0])
        return axis / np.linalg.norm(axis), angle
    axis = np.array([
        matrix[2, 1] - matrix[1, 2],
        matrix[0, 2] - matrix[2, 0],
        matrix[1, 0] - matrix[0, 1],
    ])
    return axis / np.linalg.norm(axis), angle


def _rotation(axis, angle):
    x, y, z = axis
    cross = np.array(((0, -z, y), (z, 0, -x), (-y, x, 0)), dtype=float)
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)


def _match_bones(bindings, skeleton):
    names = skeleton.get("bone_names")
    if not isinstance(names, list) or not names:
        raise ValueError("The selected rig export has no bones.")
    normalized = {}
    for index, name in enumerate(names):
        if name is None:
            continue
        normalized.setdefault(_bone_key(name), []).append(index)
    mapping = {}
    for binding in bindings:
        key = _bone_key(binding["mixamo"])
        matches = normalized.get(key, [])
        if len(matches) > 1:
            raise ValueError(f"Rig has more than one match for {binding['mixamo']}.")
        if matches:
            mapping[binding["posecode"]] = matches[0]
    required = ("pelvis", "spine", "chest", "head", "hip_left", "knee_left",
                "ankle_left", "hip_right", "knee_right", "ankle_right")
    missing = [name for name in required if name not in mapping]
    if missing:
        raise ValueError("Rig is missing required Posecode bones: " + ", ".join(missing))
    return mapping


def _bone_key(name):
    """Normalize Mixamo names with or without Blender's namespace separator."""
    plain = re.sub(r"^mixamorig\d*:?", "", name, flags=re.I)
    return semantic_name(plain).replace(" ", "")


def _euler_xyz(degrees):
    x, y, z = (math.radians(value) for value in _vector(degrees, "Euler rotation"))
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    rx = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=float)
    ry = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=float)
    rz = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=float)
    return rx @ ry @ rz


def _axis_y(radians):
    c, s = math.cos(radians), math.sin(radians)
    return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=float)


def _vector(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 3 or not all(_finite(item) for item in value):
        raise ValueError(f"Posecode {label} must contain three finite numbers.")
    return tuple(float(item) for item in value)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
