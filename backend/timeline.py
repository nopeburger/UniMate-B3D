"""Prompt chaining and exact frame-range retiming for the UniMate worker."""
import numpy as np

def native_index(clip, reference, length):
    """Generated frame (0..length-1) that a reference frame maps to within its clip."""
    return int(round((reference["frame"] - clip["start"]) /
                     (clip["end"] - clip["start"]) * (length - 1)))


def plan_windows(clips, window, overlap, extend=True):
    """Split clips into chained model windows.

    The first window of the sequence yields `window` new frames; every later
    window repeats the previous window's last `overlap` frames as context and
    yields `window - overlap` new ones. With `extend`, a clip gets as many
    windows as its duration needs, so long clips gain motion instead of being
    stretched. Returns [(clip_index, {window_slot: reference})].
    """
    step = window - overlap
    plan = []
    for index, clip in enumerate(clips):
        head = window if index == 0 else step
        duration = clip["end"] - clip["start"] + 1
        count = max(1, 1 + int(round((duration - head) / step))) if extend else 1
        length = head + step * (count - 1)
        native = {}
        for reference in clip.get("references", []):
            n = native_index(clip, reference, length)
            if n in native:
                raise ValueError("Pose references are too close together for the generated frames; move them apart.")
            native[n] = reference
        cursor = 0
        for w in range(count):
            first_slot = 0 if index == 0 and w == 0 else overlap
            new = window - first_slot
            slots = {first_slot + n - cursor: ref for n, ref in native.items() if cursor <= n < cursor + new}
            plan.append((index, slots))
            cursor += new
    return plan


def constraints(previous, slots, shape, overlap, mean, std, device):
    import torch
    if previous is None and not slots:
        return None, None
    known = torch.zeros(shape, device=device)
    mask = torch.zeros(shape, device=device, dtype=torch.bool)
    if previous is not None:
        known[..., :overlap] = previous[..., -overlap:]
        mask[..., :overlap] = True
    for slot, reference in slots.items():
        feature = np.asarray(reference["pose"]["features"], dtype=np.float32)
        if feature.shape != mean.shape or not np.isfinite(feature).all():
            raise ValueError("Invalid captured pose features.")
        normalized = (feature - mean) / std
        known[0, :len(mean), :, slot] = torch.as_tensor(normalized, device=device)
        # Pin pose and root height, while allowing trajectory velocity to be generated.
        mask[0, :len(mean), :9, slot] = True
    return known, mask

def to_local(rotations, parents):
    local = rotations.copy()
    for j, parent in enumerate(parents[1:], 1):
        local[:, j] = rotations[:, parent].swapaxes(-1, -2) @ rotations[:, j]
    return local


def forward_kinematics(root, local, skeleton):
    """Rebuild heads from fixed rest offsets, never interpolate child translations."""
    parents = skeleton["parents"]
    heads = np.asarray(skeleton["heads"])
    pos, rot = np.empty(local.shape[:2] + (3,)), np.empty_like(local)
    pos[:, 0], rot[:, 0] = root, local[:, 0]
    for j, parent in enumerate(parents[1:], 1):
        rot[:, j] = rot[:, parent] @ local[:, j]
        pos[:, j] = pos[:, parent] + np.einsum("tij,j->ti", rot[:, parent], heads[j]-heads[parent])
    return pos, rot


def ease(u):
    return u*u*u*(10 + u*(-15 + 6*u))


def limit_vectors(vectors, limit):
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors * np.minimum(1., limit / np.maximum(norms, 1e-12))


def pose_bridge(root, local, start, target, incoming=True):
    from scipy.spatial.transform import Rotation
    n = target-start
    start_rot, target_rot = local[start].copy(), local[target].copy()
    start_root, target_root = root[start].copy(), root[target].copy()
    delta = Rotation.from_matrix(start_rot.swapaxes(-1,-2) @ target_rot).as_rotvec()
    # Bounded incoming velocity adds follow-through without large overshoots.
    if incoming and start > 0:
        tangent = Rotation.from_matrix(local[start-1].swapaxes(-1,-2) @ start_rot).as_rotvec()*n
        tangent = limit_vectors(tangent, .5)
        # Root limit scales with this rig, so miniature and large rigs behave alike.
        distance = np.linalg.norm(target_root-start_root)
        root_tangent = limit_vectors((root[start]-root[start-1])*n, max(distance*.4, 1e-8))
    else:
        tangent, root_tangent = np.zeros_like(delta), np.zeros(3)
    for t in range(1, n+1):
        u = t/n
        weight = ease(u)
        h = u-6*u**3+8*u**4-3*u**5
        local[start+t] = start_rot @ Rotation.from_rotvec(weight*delta+h*tangent).as_matrix()
        root[start+t] = start_root + weight*(target_root-start_root) + h*root_tangent


def inertialize(root, local, cursor, end):
    """Blend frames cursor..end-1 so motion continues from frame cursor-1.

    Extrapolates the preceding motion one frame and decays only the position
    and velocity gaps, for the root and every joint rotation. Adding the whole
    preceding velocity instead of the gap double-counted motion (the hips
    lurched to old + new speed, then stopped hard); snapping to the previous
    pose froze every joint for a frame.
    """
    from scipy.spatial.transform import Rotation
    n = end-cursor
    if cursor < 1 or n < 1 or cursor+1 >= len(root):
        return
    def step(a, b):  # per-joint rotation taking b to a, as rotation vectors
        return Rotation.from_matrix((a @ b.swapaxes(-1,-2)).reshape(-1,3,3)).as_rotvec()
    previous_spin = step(local[cursor-1], local[cursor-2]) if cursor > 1 else np.zeros((local.shape[1], 3))
    extrapolated = Rotation.from_rotvec(previous_spin).as_matrix() @ local[cursor-1]
    rot_error = step(extrapolated, local[cursor])
    spin_error = previous_spin-step(local[cursor+1], local[cursor])
    previous_velocity = root[cursor-1]-root[cursor-2] if cursor > 1 else np.zeros(3)
    root_error = root[cursor-1]+previous_velocity-root[cursor]
    velocity_error = previous_velocity-(root[cursor+1]-root[cursor])
    for t in range(n):
        weight = 1-ease(t/n)
        local[cursor+t] = Rotation.from_rotvec((rot_error+spin_error*t)*weight).as_matrix() @ local[cursor+t]
        root[cursor+t] += (root_error + velocity_error*t)*weight


def finish_motion(root, local, clips, skeleton, transition_frames, pose_approach_frames, window_seams=()):
    """Deterministic editing of neural motion; reference approaches are pose blends.

    Joins (between clips, and between chained model windows inside a clip)
    are inertialized. Each captured reference is approached and left with
    pose blends, so it stays exact.
    """
    cursor = 0
    for clip in clips:
        count = clip["end"]-clip["start"]+1
        refs = sorted(clip.get("references", []), key=lambda r: r["frame"])
        targets = [cursor+ref["frame"]-clip["start"] for ref in refs]
        # Joins never blend across a reference, so captured targets stay exact.
        joins = ([cursor] if cursor else []) + [seam for seam in window_seams if cursor < seam < cursor+count]
        for join in joins:
            end = min([join+transition_frames, cursor+count-1] + [t for t in targets if t >= join])
            inertialize(root, local, join, end)
        # Each reference is approached from an earlier pose with zero endpoint
        # speed; local rotations avoid collapsing limbs as their parents turn.
        previous_ref = cursor-1 if cursor else 0
        starts = []
        for target in targets:
            start = max(previous_ref, target-pose_approach_frames)
            starts.append(start)
            if pose_approach_frames and target > start:
                pose_bridge(root, local, start, target)
            previous_ref = target
        # Every reference is also left smoothly, up to where the next approach
        # begins. Only releasing the last one left a pop after earlier ones.
        if pose_approach_frames and transition_frames:
            for index, target in enumerate(targets):
                limit = starts[index+1] if index+1 < len(targets) else cursor+count-1
                release = min(target+transition_frames, limit)
                if release > target:
                    pose_bridge(root, local, target, release, incoming=False)
        cursor += count
    return forward_kinematics(root, local, skeleton)


def retime(positions, rotations, spans, clips, skeleton,
           transition_frames=12, pose_approach_frames=60, seams=None):
    """Resample each clip's generated frames (spans) onto its frame range.

    seams lists, per clip, the generated-frame offsets (from the clip's span
    start) where a chained model window begins; those joins are smoothed too.
    """
    from scipy.spatial.transform import Rotation, Slerp
    if not 0 <= transition_frames <= 120 or not 0 <= pose_approach_frames <= 600:
        raise ValueError("Invalid transition or pose approach duration.")
    local = to_local(rotations, skeleton["parents"])
    all_root, all_local, window_seams = [], [], []
    cursor = 0
    for index, ((begin, end), clip) in enumerate(zip(spans, clips)):
        root, rot = positions[begin:end, 0], local[begin:end]
        duration = clip["end"] - clip["start"] + 1
        knots = {0: 0, duration-1: len(root)-1}
        for ref in clip.get("references", []):
            knots[ref["frame"] - clip["start"]] = native_index(clip, ref, len(root))
        destinations = np.array(sorted(knots))
        sources = np.array([knots[i] for i in destinations])
        if np.any(np.diff(sources) <= 0):
            raise ValueError("Move pose references farther apart within the prompt clip.")
        sample_times = np.interp(np.arange(duration), destinations, sources)
        low = np.floor(sample_times).astype(int)
        high = np.minimum(low+1, len(root)-1)
        weights = (sample_times-low)[:, None]
        all_root.append(root[low]*(1-weights)+root[high]*weights)
        output = np.empty((duration, rot.shape[1], 3, 3))
        for joint in range(rot.shape[1]):
            output[:, joint] = Slerp(np.arange(len(root)), Rotation.from_matrix(rot[:,joint]))(sample_times).as_matrix()
        all_local.append(output)
        for seam in (seams[index] if seams else []):
            window_seams.append(cursor + int(np.searchsorted(sample_times, seam)))
        cursor += duration
    return finish_motion(np.concatenate(all_root), np.concatenate(all_local), clips,
                         skeleton, transition_frames, pose_approach_frames, window_seams)
