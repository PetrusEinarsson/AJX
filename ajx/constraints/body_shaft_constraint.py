from abc import ABC, abstractmethod

import jax
import ajx.math as math
from jax import jit
import jax.numpy as jnp

from enum import Enum
from ajx.definitions import Transform, State
from ajx.param import SimulationParameters
from typing import Union, Tuple
from functools import partial
from ajx.constraints.base import Constraint, ConstraintType, get_frame_transform


@jit
def get_frame_transform(frame, constraint_id, body_pos, body_rotation):
    d0 = frame.position[constraint_id]
    frame_rot0 = frame.rotation[constraint_id]
    frame_rot = math.quat_mul(body_rotation, frame_rot0)
    d = math.rotate_vector(body_rotation, d0)
    u = math.rotation_matrix(frame_rot)[:, 0]
    v = math.rotation_matrix(frame_rot)[:, 1]
    w = math.rotation_matrix(frame_rot)[:, 2]

    return d, u, v, w


class BodyShaftConstraint(Constraint):
    """
    A constraint that restricts the relative motion between two bodies.

    Constraint types currently supported:
     - Hinge
     - Primsatic
    """

    name: str
    body: str
    scalar_body: str
    constraint_type: ConstraintType
    is_attached_to_world: bool = False

    def get_num_bodies():
        return 2

    @property
    def bodies(self):
        return (self.body, self.scalar_body)

    def __init__(
        self,
        name: str,
        body: str,
        scalar_body: str,
        constraint_type: ConstraintType,
    ):
        self.name = name
        self.body = body
        self.scalar_body = scalar_body
        self.constraint_type = constraint_type

    def get_multiplier_names(self) -> Tuple[str]:
        if self.constraint_type == ConstraintType.HINGE.value:
            return ("nx", "ny", "nz", "n_bend", "n_torsion", "t")
        elif self.constraint_type == ConstraintType.PRISMATIC.value:
            return ("nu", "nw", "n_bend1", "n_torsion", "n_bend2", "t")
        return ()

    @partial(jit, static_argnums=0)
    def func(
        self,
        param: SimulationParameters,
        state: State,
    ):
        body_id = param.rigid_body_param.names.index(self.body)
        scalar_body_id = param.scalar_body_param.names.index(self.scalar_body)
        constraint_id = param.constraint_param.names.index(self.name)
        return BodyShaftConstraint.func(
            param,
            state,
            (body_id, scalar_body_id),
            constraint_id,
            self.constraint_type,
        )

    @jit
    def func(
        param: SimulationParameters,
        state: State,
        body_ids: Tuple[Union[int, jax.Array]],
        constraint_id: Union[int, jax.Array],
        constraint_type: Union[ConstraintType, jax.Array],
    ) -> jax.Array:
        """
        C
        """
        body_id = body_ids[0]
        scalar_body_id = body_ids[1]
        scalar_body_pos = state.conf.scalar[scalar_body_id]
        body_pos = state.conf.pos[body_id]
        body_rot = state.conf.rot[body_id]
        world_pos = jnp.array([0.0, 0.0, 0.0])
        world_rot = jnp.array([1.0, 0.0, 0.0, 0.0])

        d_a, u_a, v_a, w_a = get_frame_transform(
            param.constraint_param.frame_a, constraint_id, world_pos, world_rot
        )

        d_b, u_b, v_b, w_b = get_frame_transform(
            param.constraint_param.frame_b, constraint_id, body_pos, body_rot
        )

        r_a = world_pos + d_a
        r_b = body_pos + d_b

        # TODO: Copy and paste code. Computations are done twice...
        frame_a_pos0 = param.constraint_param.frame_a.position[constraint_id]
        frame_a_rot0 = param.constraint_param.frame_a.rotation[constraint_id]
        frame_a_rot = math.quat_mul(world_rot, frame_a_rot0)
        d_a = math.rotate_vector(world_rot, frame_a_pos0)

        frame_b_pos0 = param.constraint_param.frame_b.position[constraint_id]
        frame_b_rot0 = param.constraint_param.frame_b.rotation[constraint_id]
        frame_b_rot = math.quat_mul(body_rot, frame_b_rot0)
        d_b = math.rotate_vector(body_rot, frame_b_pos0)

        # For prismatic
        r_a = world_pos + d_a
        r_b = body_pos + d_b
        r_delta = r_a - r_b
        v_a = math.rotation_matrix(frame_a_rot)[:, 1]
        free_prismatic = jnp.dot(v_a, r_delta) - scalar_body_pos

        # For hinge
        frame_a_rot_inv = math.conjugate(frame_a_rot)
        delta_rotation = math.quat_mul(frame_a_rot_inv, frame_b_rot)
        axis_angle = math.to_rotation_vector(delta_rotation)
        axis = jnp.array([1.0, 0.0, 0.0])
        free_hinge = -jnp.dot(axis_angle, axis) - scalar_body_pos

        hinge_constraint = free_hinge[None] * (
            constraint_type == ConstraintType.HINGE.value
        )
        prismatic_constraint = free_prismatic[None] * (
            constraint_type == ConstraintType.PRISMATIC.value
        )
        return hinge_constraint + prismatic_constraint

    @jit
    def jacobian(
        param: SimulationParameters,
        state: State,
        body_ids: Tuple[Union[int, jax.Array]],
        constraint_id: Union[int, jax.Array],
        constraint_type: Union[ConstraintType, jax.Array],
    ) -> jax.Array:
        body_id = body_ids[0]
        scalar_body_id = body_ids[1]
        scalar_body_pos = state.conf.scalar[scalar_body_id]
        body_pos = state.conf.pos[body_id]
        body_rot = state.conf.rot[body_id]
        world_pos = jnp.array([0.0, 0.0, 0.0])
        world_rot = jnp.array([1.0, 0.0, 0.0, 0.0])

        d_a, u_a, v_a, w_a = get_frame_transform(
            param.constraint_param.frame_a, constraint_id, world_pos, world_rot
        )
        d_b, u_b, v_b, w_b = get_frame_transform(
            param.constraint_param.frame_b, constraint_id, body_pos, body_rot
        )

        r_a = world_pos + d_a
        r_b = body_pos + d_b

        dot2_3_b = jnp.block([-v_a[None], jnp.zeros([1, 3])])
        u_a_tangent_b = jnp.block([jnp.zeros([1, 3]), -u_a])

        jac_hinge = jnp.stack(
            [
                jnp.concatenate([u_a_tangent_b, jnp.array([-1.0])]),
            ]
        ) * (constraint_type == 0)
        jac_prismatic = jnp.stack(
            [
                jnp.concatenate([dot2_3_b, jnp.array([-1.0])]),
            ]
        ) * (constraint_type == 1)

        return jac_hinge + jac_prismatic
