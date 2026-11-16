"""
The window frame: motion relative to where a person stands and faces at one moment.

Origin: the pelvis at the reference frame (t = 0, the last history frame). Axes as in ROS
(REP 103): x forward, y left, z up. The heading comes from the hip line, the pelvis's left-right
axis, which stays horizontal when the person bends forward. Only the rotation about z is removed,
so z stays gravity and lean is kept.

Every function takes numpy arrays in a frame with z up.
"""
import numpy as np


def heading_frame(hip_line):
    """
    Rotation from the world into the window frame, from the hip line.

    :param hip_line: (..., 3) vector pointing to the person's left, any length.
    :return: rotation (..., 3, 3), rows forward, left, up; and length (...,), the projection of the
        unit hip line onto the floor: 1 when level, 0 when vertical. Below MIN_HIP_LINE_LENGTH
        (interact/constants.py) the heading is unreliable; at 0 the rotation has zero rows.
    """
    hip_line = np.asarray(hip_line, dtype=np.float64)
    left = hip_line / np.linalg.norm(hip_line, axis=-1, keepdims=True)
    left[..., 2] = 0
    length = np.linalg.norm(left, axis=-1)
    left /= np.maximum(length, np.finfo(np.float64).tiny)[..., None]
    # forward = left x up
    forward = np.stack([left[..., 1], -left[..., 0], np.zeros_like(length)], axis=-1)
    up = np.broadcast_to(np.array([0.0, 0.0, 1.0]), forward.shape)
    return np.stack([forward, left, up], axis=-2), length


def to_frame(points, origin, rotation):
    """
    World -> window frame.

    :param points: (..., 3) in the world.
    :param origin: (3,) pelvis position at the reference frame.
    :param rotation: (3, 3) from heading_frame.
    :return: (..., 3) in the window frame, dtype of points.
    """
    points = np.asarray(points)
    return ((points - origin) @ np.asarray(rotation).T).astype(points.dtype, copy=False)


def from_frame(points, origin, rotation):
    """Window frame -> world, the inverse of to_frame."""
    points = np.asarray(points)
    return (points @ np.asarray(rotation) + origin).astype(points.dtype, copy=False)
