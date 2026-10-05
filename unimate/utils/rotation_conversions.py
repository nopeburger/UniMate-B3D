"""6D rotation representation → rotation matrix conversions.

The 6D representation is from Zhou et al., "On the Continuity of Rotation
Representations in Neural Networks", CVPR 2019
(http://arxiv.org/abs/1812.07035): the first two rows of the rotation matrix,
orthonormalized via Gram-Schmidt on decode.

Convention: rotation matrices act on column vectors via post-multiplication,
i.e. ``transformed_point = R @ point``.
"""

import numpy as np
import torch


def rotation_6d_to_matrix_safe(cont6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation to matrix with NaN protection and epsilon offset.

    Args:
        cont6d: 6D rotation representation (*, 6).

    Returns:
        Rotation matrices (*, 3, 3).
    """
    assert cont6d.shape[-1] == 6, "The last dimension must be 6"
    epsilon = 1e-8
    cont6d = torch.nan_to_num(cont6d) + epsilon
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]
    x = x_raw / torch.linalg.norm(x_raw, dim=-1, keepdims=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.linalg.norm(z, dim=-1, keepdims=True)
    y = torch.cross(z, x, dim=-1)
    return torch.cat([x[..., None], y[..., None], z[..., None]], dim=-1)


def rotation_6d_to_matrix_np(cont6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation to matrix (numpy version).

    Args:
        cont6d: 6D rotation representation (*, 6).

    Returns:
        Rotation matrices (*, 3, 3).
    """
    assert cont6d.shape[-1] == 6, "The last dimension must be 6"
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]
    x = x_raw / np.linalg.norm(x_raw, axis=-1, keepdims=True)
    z = np.cross(x, y_raw, axis=-1)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x, axis=-1)
    return np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)


def matrix_to_quaternion_np(mats: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to unit quaternions (numpy version).

    Shepperd's method with one branch per matrix, chosen by the largest of
    ``|w|, |x|, |y|, |z|``. ``Quaternions.from_transforms`` (Motion library)
    applies every branch whose candidate ties for the largest, so a rotation
    with two equal components (e.g. exactly 90 deg about one axis) comes back
    as a different rotation; use this instead.

    Args:
        mats: Rotation matrices (*, 3, 3).

    Returns:
        Quaternions (*, 4) as ``(w, x, y, z)``, ``w >= 0``.
    """
    m = np.asarray(mats, dtype=np.float64)
    d0, d1, d2 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    cand = np.stack([1 + d0 + d1 + d2, 1 + d0 - d1 - d2,
                     1 - d0 + d1 - d2, 1 - d0 - d1 + d2], axis=-1)   # 4 * component^2
    branch = np.argmax(cand, axis=-1)
    s = 2.0 * np.sqrt(np.maximum(np.take_along_axis(cand, branch[..., None], -1)[..., 0], 1e-12))
    m21_m12, m02_m20, m10_m01 = m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]
    m21p12, m02p20, m10p01 = m[..., 2, 1] + m[..., 1, 2], m[..., 0, 2] + m[..., 2, 0], m[..., 1, 0] + m[..., 0, 1]
    q = np.where(branch[..., None] == 0, np.stack([s / 4, m21_m12 / s, m02_m20 / s, m10_m01 / s], -1),
        np.where(branch[..., None] == 1, np.stack([m21_m12 / s, s / 4, m10p01 / s, m02p20 / s], -1),
        np.where(branch[..., None] == 2, np.stack([m02_m20 / s, m10p01 / s, s / 4, m21p12 / s], -1),
                 np.stack([m10_m01 / s, m02p20 / s, m21p12 / s, s / 4], -1))))
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(q[..., :1] < 0, -q, q)
