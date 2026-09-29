"""Human hand coordinate representations and controlled global-frame changes."""

import numpy as np


HUMAN_REPRESENTATIONS = ("base_xyz", "wrist_relative", "palm_relative")


def validate_global_transform(transform) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1e-5)):
        raise ValueError("Human global transform must be a proper 4x4 rigid transform")
    return matrix


def hand_xyz_features(points: np.ndarray, observed: np.ndarray,
                      representation: str, first_palm_pose: np.ndarray | None = None,
                      global_transform: np.ndarray | None = None) -> np.ndarray:
    """Encode measured base-frame points; missing frames remain exactly zero."""
    if representation not in HUMAN_REPRESENTATIONS:
        raise ValueError(f"Unknown human representation: {representation}")
    points = np.asarray(points)
    observed = np.asarray(observed, dtype=bool)
    if points.ndim != 3 or points.shape[1:] != (21, 3) or observed.shape != (len(points),):
        raise ValueError("Expected 21x3 hand points and aligned frame observations")
    valid_frames = np.flatnonzero(observed)
    if len(valid_frames) < 2 or not np.isfinite(points[observed]).all():
        raise ValueError("At least two finite hand observations are required")

    transform = (validate_global_transform(global_transform)
                 if global_transform is not None else None)
    encoded = np.zeros(points.shape, dtype=np.float64)
    encoded[observed] = points[observed]
    if transform is not None:
        encoded[observed] = encoded[observed] @ transform[:3, :3].T + transform[:3, 3]

    first = int(valid_frames[0])
    if representation == "wrist_relative":
        encoded[observed] -= encoded[first, 0]
    elif representation == "palm_relative":
        if first_palm_pose is None:
            raise ValueError("palm_relative requires a measured first palm pose")
        palm = validate_global_transform(first_palm_pose)
        if transform is not None:
            palm = transform @ palm
        encoded[observed] = (encoded[observed] - palm[:3, 3]) @ palm[:3, :3]
    return encoded.astype(np.float32)
