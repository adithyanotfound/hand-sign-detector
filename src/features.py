"""Shared hand-landmark feature code, used by both training and live inference.

Keeping this in one place guarantees the model sees identical features in both.
"""
import numpy as np

NUM_LANDMARKS = 21
WRIST, MIDDLE_MCP = 0, 9

# MediaPipe's hand skeleton, used for drawing.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (5, 9), (9, 10), (10, 11), (11, 12),    # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20), # pinky
    (0, 17),
]


def to_isotropic(normalized, width, height):
    """Convert MediaPipe's 0-1 coordinates to a space where x, y and z share a scale.

    x and y are normalised by image width and height separately, so without this a
    hand's proportions would change with the camera's aspect ratio. MediaPipe's z
    uses roughly the same scale as x.
    """
    points = np.asarray(normalized, dtype=np.float32).copy()
    points[:, 0] *= width
    points[:, 1] *= height
    points[:, 2] *= width
    return points


def featurize(points):
    """Turn 21x3 isotropic landmarks into a position- and scale-invariant vector."""
    points = points - points[WRIST]
    scale = np.linalg.norm(points[:, :2], axis=1).max()
    if scale < 1e-6:
        return np.zeros(NUM_LANDMARKS * 3, dtype=np.float32)
    return (points / scale).astype(np.float32).ravel()
