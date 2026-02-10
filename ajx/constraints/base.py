from abc import ABC, abstractmethod

import ajx.math as math
from jax import jit
import jax.numpy as jnp

from enum import Enum


class ConstraintType(Enum):
    HINGE = 0
    PRISMATIC = 1


@jit
def get_frame_transform(frame, body_pos, body_rotation):
    d0 = frame.position
    frame_rot0 = frame.rotation
    frame_rot = math.quat_mul(body_rotation, frame_rot0)
    d = math.rotate_vector(body_rotation, d0)
    u = math.rotation_matrix(frame_rot)[:, 0]
    v = math.rotation_matrix(frame_rot)[:, 1]
    w = math.rotation_matrix(frame_rot)[:, 2]

    return d, u, v, w


class Constraint(ABC):
    pass
