from abc import ABC, abstractmethod

import ajx.math as math
from jax import jit
import jax.numpy as jnp


class Constraint(ABC):
    @abstractmethod
    def c(self, param, state):
        pass

    @abstractmethod
    def jac(self, param, state):
        pass

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

    @jit
    def multi_constraint_func6x2(
        param,
        state,
        body_a_id,
        body_b_id,
        constraint_id,
        constraint_type,
    ):
        body_b_pos = state.conf.pos[body_b_id]
        body_b_rot = state.conf.rot[body_b_id]
        body_a_pos = state.conf.pos[body_a_id]
        body_a_rot = state.conf.rot[body_a_id]

        d_a, u_a, v_a, w_a = Constraint.get_frame_transform(
            param.constraint_param[constraint_id].frame_a, body_a_pos, body_a_rot
        )

        d_b, u_b, v_b, w_b = Constraint.get_frame_transform(
            param.constraint_param[constraint_id].frame_b, body_b_pos, body_b_rot
        )

        r_a = body_a_pos + d_a
        r_b = body_b_pos + d_b

        spherical = r_a - r_b
        dot1_1 = jnp.dot(u_a, v_b)
        dot1_2 = jnp.dot(u_a, w_b)
        dot1_3 = jnp.dot(w_a, v_b)
        dot2_1 = jnp.dot(u_a, r_a - r_b)
        dot2_2 = jnp.dot(w_a, r_a - r_b)

        # TODO: Copy and paste code. Computations are done twice...
        frame_a_pos0 = param.constraint_param[constraint_id].frame_a.position
        frame_a_rot0 = param.constraint_param[constraint_id].frame_a.rotation
        frame_a_rot = math.quat_mul(body_a_rot, frame_a_rot0)
        d_a = math.rotate_vector(body_a_rot, frame_a_pos0)

        frame_b_pos0 = param.constraint_param[constraint_id].frame_b.position
        frame_b_rot0 = param.constraint_param[constraint_id].frame_b.rotation
        frame_b_rot = math.quat_mul(body_b_rot, frame_b_rot0)
        d_b = math.rotate_vector(body_b_rot, frame_b_pos0)

        # For prismatic
        r_a = body_a_pos + d_a
        r_b = body_b_pos + d_b
        r_delta = r_a - r_b
        v_a = math.rotation_matrix(frame_a_rot)[:, 1]
        free_prismatic = jnp.dot(v_a, r_delta)

        # For hinge
        frame_a_rot_inv = math.conjugate(frame_a_rot)
        delta_rotation = math.quat_mul(frame_a_rot_inv, frame_b_rot)
        axis_angle = math.to_rotation_vector(delta_rotation)
        axis = jnp.array([1.0, 0.0, 0.0])
        free_hinge = -jnp.dot(axis_angle, axis)

        hinge_constraint = jnp.block(
            [spherical, dot1_1[None], dot1_2[None], free_hinge[None]]
        ) * (constraint_type == 0)
        prismatic_constraint = jnp.block(
            [dot2_1, dot2_2, dot1_1, dot1_2, dot1_3, free_prismatic[None]]
        ) * (constraint_type == 1)
        return hinge_constraint + prismatic_constraint

    @jit
    def multi_free_degree6x1(
        param,
        state,
        body_a_id,
        body_b_id,
        constraint_id,
        constraint_type,
    ):
        body_b_pos = state.conf.pos[body_b_id]
        body_b_rot = state.conf.rot[body_b_id]
        body_a_pos = jnp.array([0.0, 0.0, 0.0])
        body_a_rot = jnp.array([1.0, 0.0, 0.0, 0.0])

        frame_a_pos0 = param.constraint_param[constraint_id].frame_a.position
        frame_a_rot0 = param.constraint_param[constraint_id].frame_a.rotation
        frame_a_rot = math.quat_mul(body_a_rot, frame_a_rot0)
        d_a = math.rotate_vector(body_a_rot, frame_a_pos0)

        frame_b_pos0 = param.constraint_param[constraint_id].frame_b.position
        frame_b_rot0 = param.constraint_param[constraint_id].frame_b.rotation
        frame_b_rot = math.quat_mul(body_b_rot, frame_b_rot0)
        d_b = math.rotate_vector(body_b_rot, frame_b_pos0)

        # For prismatic
        r_a = body_a_pos + d_a
        r_b = body_b_pos + d_b
        r_delta = r_a - r_b
        v_a = math.rotation_matrix(frame_a_rot)[:, 1]
        x = jnp.dot(v_a, r_delta)
        free_prismatic = x * (constraint_type == 1)

        # For hinge
        frame_a_rot_inv = math.conjugate(frame_a_rot)
        delta_rotation = math.quat_mul(frame_a_rot_inv, frame_b_rot)
        axis_angle = math.to_rotation_vector(delta_rotation)
        axis = jnp.array([1.0, 0.0, 0.0])
        theta = jnp.dot(axis_angle, axis)
        free_hinge = theta * (constraint_type == 0)
        return free_hinge + free_prismatic

    @jit
    def multi_free_degree6x2(
        param,
        state,
        body_a_id,
        body_b_id,
        constraint_id,
        constraint_type,
    ):
        body_b_pos = state.conf.pos[body_b_id]
        body_b_rot = state.conf.rot[body_b_id]
        body_a_pos = state.conf.pos[body_a_id]
        body_a_rot = state.conf.rot[body_a_id]

        frame_a_pos0 = param.constraint_param[constraint_id].frame_a.position
        frame_a_rot0 = param.constraint_param[constraint_id].frame_a.rotation
        frame_a_rot = math.quat_mul(body_a_rot, frame_a_rot0)
        d_a = math.rotate_vector(body_a_rot, frame_a_pos0)

        frame_b_pos0 = param.constraint_param[constraint_id].frame_b.position
        frame_b_rot0 = param.constraint_param[constraint_id].frame_b.rotation
        frame_b_rot = math.quat_mul(body_b_rot, frame_b_rot0)
        d_b = math.rotate_vector(body_b_rot, frame_b_pos0)

        # For prismatic
        r_a = body_a_pos + d_a
        r_b = body_b_pos + d_b
        r_delta = r_a - r_b
        v_a = math.rotation_matrix(frame_a_rot)[:, 1]
        x = jnp.dot(v_a, r_delta)
        free_prismatic = x * (constraint_type == 1)

        # For hinge
        frame_a_rot_inv = math.conjugate(frame_a_rot)
        delta_rotation = math.quat_mul(frame_a_rot_inv, frame_b_rot)
        axis_angle = math.to_rotation_vector(delta_rotation)
        axis = jnp.array([1.0, 0.0, 0.0])
        theta = -jnp.dot(axis_angle, axis)
        free_hinge = theta * (constraint_type == 0)
        return free_hinge + free_prismatic

    @jit
    def multi_jacobian6x2(
        param,
        state,
        body_a_id,
        body_b_id,
        constraint_id,
        constraint_type,
    ):
        body_b_pos = state.conf.pos[body_b_id]
        body_b_rot = state.conf.rot[body_b_id]
        body_a_pos = state.conf.pos[body_a_id]
        body_a_rot = state.conf.rot[body_a_id]

        d_a, u_a, v_a, w_a = Constraint.get_frame_transform(
            param.constraint_param[constraint_id].frame_a, body_a_pos, body_a_rot
        )
        d_b, u_b, v_b, w_b = Constraint.get_frame_transform(
            param.constraint_param[constraint_id].frame_b, body_b_pos, body_b_rot
        )

        r_a = body_a_pos + d_a
        r_b = body_b_pos + d_b

        spherical_a = jnp.block([jnp.eye(3), -math.skew(d_a)])
        spherical_b = jnp.block([-jnp.eye(3), math.skew(d_b)])
        dot1_1_a = jnp.block([jnp.zeros([1, 3]), jnp.cross(u_a, v_b)])
        dot1_1_b = jnp.block([jnp.zeros([1, 3]), -jnp.cross(u_a, v_b)])
        dot1_2_a = jnp.block([jnp.zeros([1, 3]), jnp.cross(u_a, w_b)])
        dot1_2_b = jnp.block([jnp.zeros([1, 3]), -jnp.cross(u_a, w_b)])
        dot1_3_a = jnp.block([jnp.zeros([1, 3]), jnp.cross(w_a, v_b)])
        dot1_3_b = jnp.block([jnp.zeros([1, 3]), -jnp.cross(w_a, v_b)])

        dot2_1_a = jnp.block([u_a[None], jnp.cross(u_a, r_a - r_b)])
        dot2_1_b = jnp.block([-u_a[None], jnp.zeros([1, 3])])
        dot2_2_a = jnp.block([w_a[None], jnp.cross(w_a, r_a - r_b)])
        dot2_2_b = jnp.block([-w_a[None], jnp.zeros([1, 3])])
        dot2_3_a = jnp.block([v_a[None], jnp.cross(v_a, r_a - r_b)])
        dot2_3_b = jnp.block([-v_a[None], jnp.zeros([1, 3])])
        u_a_tangent_a = jnp.block([jnp.zeros([1, 3]), u_a])
        u_a_tangent_b = jnp.block([jnp.zeros([1, 3]), -u_a])

        jac_hinge = jnp.stack(
            [
                jnp.concatenate([spherical_a, dot1_1_a, dot1_2_a, u_a_tangent_a]),
                jnp.concatenate([spherical_b, dot1_1_b, dot1_2_b, u_a_tangent_b]),
            ]
        ) * (constraint_type == 0)
        jac_prismatic = jnp.stack(
            [
                jnp.concatenate(
                    [dot2_1_a, dot2_2_a, dot1_1_a, dot1_2_a, dot1_3_a, dot2_3_a]
                ),
                jnp.concatenate(
                    [dot2_1_b, dot2_2_b, dot1_1_b, dot1_2_b, dot1_3_b, dot2_3_b]
                ),
            ]
        ) * (constraint_type == 1)

        return jac_hinge + jac_prismatic

    @jit
    def multi_constraint_func6x1(
        param,
        state,
        body_a_id,
        body_b_id,
        constraint_id,
        constraint_type,
    ):
        body_b_pos = state.conf.pos[body_b_id]
        body_b_rot = state.conf.rot[body_b_id]

        body_a_pos = jnp.array([0.0, 0.0, 0.0])
        body_a_rot = jnp.array([1.0, 0.0, 0.0, 0.0])

        d_a, u_a, v_a, w_a = Constraint.get_frame_transform(
            param.constraint_param[constraint_id].frame_a, body_a_pos, body_a_rot
        )

        d_b, u_b, v_b, w_b = Constraint.get_frame_transform(
            param.constraint_param[constraint_id].frame_b, body_b_pos, body_b_rot
        )

        r_a = body_a_pos + d_a
        r_b = body_b_pos + d_b

        spherical = r_a - r_b
        dot1_1 = jnp.dot(u_a, v_b)
        dot1_2 = jnp.dot(u_a, w_b)
        dot1_3 = jnp.dot(w_a, v_b)
        dot2_1 = jnp.dot(u_a, r_a - r_b)
        dot2_2 = jnp.dot(w_a, r_a - r_b)

        # TODO: Copy and paste code. Computations are done twice...
        frame_a_pos0 = param.constraint_param[constraint_id].frame_a.position
        frame_a_rot0 = param.constraint_param[constraint_id].frame_a.rotation
        frame_a_rot = math.quat_mul(body_a_rot, frame_a_rot0)
        d_a = math.rotate_vector(body_a_rot, frame_a_pos0)

        frame_b_pos0 = param.constraint_param[constraint_id].frame_b.position
        frame_b_rot0 = param.constraint_param[constraint_id].frame_b.rotation
        frame_b_rot = math.quat_mul(body_b_rot, frame_b_rot0)
        d_b = math.rotate_vector(body_b_rot, frame_b_pos0)

        # For prismatic
        r_a = body_a_pos + d_a
        r_b = body_b_pos + d_b
        r_delta = r_a - r_b
        v_a = math.rotation_matrix(frame_a_rot)[:, 1]
        free_prismatic = jnp.dot(v_a, r_delta)

        # For hinge
        frame_a_rot_inv = math.conjugate(frame_a_rot)
        delta_rotation = math.quat_mul(frame_a_rot_inv, frame_b_rot)
        axis_angle = math.to_rotation_vector(delta_rotation)
        axis = jnp.array([1.0, 0.0, 0.0])
        free_hinge = -jnp.dot(axis_angle, axis)

        hinge_constraint = jnp.block(
            [spherical, dot1_1[None], dot1_2[None], free_hinge[None]]
        ) * (constraint_type == 0)
        prismatic_constraint = jnp.block(
            [dot2_1, dot2_2, dot1_1, dot1_2, dot1_3, free_prismatic[None]]
        ) * (constraint_type == 1)
        return hinge_constraint + prismatic_constraint

    @jit
    def multi_jacobian6x1(
        param,
        state,
        body_a_id,
        body_b_id,
        constraint_id,
        constraint_type,
    ):
        body_b_pos = state.conf.pos[body_b_id]
        body_b_rot = state.conf.rot[body_b_id]

        body_a_pos = jnp.array([0.0, 0.0, 0.0])
        body_a_rot = jnp.array([1.0, 0.0, 0.0, 0.0])

        d_a, u_a, v_a, w_a = Constraint.get_frame_transform(
            param.constraint_param[constraint_id].frame_a, body_a_pos, body_a_rot
        )
        d_b, u_b, v_b, w_b = Constraint.get_frame_transform(
            param.constraint_param[constraint_id].frame_b, body_b_pos, body_b_rot
        )

        spherical_b = jnp.block([-jnp.eye(3), math.skew(d_b)])
        dot1_1_b = jnp.block([jnp.zeros([1, 3]), -jnp.cross(u_a, v_b)])
        dot1_2_b = jnp.block([jnp.zeros([1, 3]), -jnp.cross(u_a, w_b)])
        dot1_3_b = jnp.block([jnp.zeros([1, 3]), -jnp.cross(w_a, v_b)])

        dot2_1_b = jnp.block([-u_a[None], jnp.zeros([1, 3])])
        dot2_2_b = jnp.block([-w_a[None], jnp.zeros([1, 3])])
        dot2_3_b = jnp.block([-v_a[None], jnp.zeros([1, 3])])
        u_a_tangent_b = jnp.block([jnp.zeros([1, 3]), -u_a])

        jac_hinge = jnp.stack(
            [
                jnp.concatenate([spherical_b, dot1_1_b, dot1_2_b, u_a_tangent_b]),
            ]
        ) * (constraint_type == 0)
        jac_prismatic = jnp.stack(
            [
                jnp.concatenate(
                    [dot2_1_b, dot2_2_b, dot1_1_b, dot1_2_b, dot1_3_b, dot2_3_b]
                ),
            ]
        ) * (constraint_type == 1)

        return jac_hinge + jac_prismatic

    @jit
    def multi_tangential6x2(
        param,
        state,
        body_a_id,
        body_b_id,
        constraint_id,
        constraint_type,
    ):
        body_b_pos = state.conf.pos[body_b_id]
        body_b_rot = state.conf.rot[body_b_id]
        body_a_pos = state.conf.pos[body_a_id]
        body_a_rot = state.conf.rot[body_a_id]

        d_a, u_a, v_a, w_a = Constraint.get_frame_transform(
            param.constraint_param[constraint_id].frame_a, body_a_pos, body_a_rot
        )
        d_b, u_b, v_b, w_b = Constraint.get_frame_transform(
            param.constraint_param[constraint_id].frame_b, body_b_pos, body_b_rot
        )

        r_a = body_a_pos + d_a
        r_b = body_b_pos + d_b

        dot2_3_a = jnp.block([v_a[None], jnp.cross(v_a, r_a - r_b)])
        dot2_3_b = jnp.block([-v_a[None], jnp.zeros([1, 3])])
        u_a_tangent_a = jnp.block([jnp.zeros([1, 3]), u_a])
        u_a_tangent_b = jnp.block([jnp.zeros([1, 3]), -u_a])

        jac_hinge = jnp.stack(
            [
                jnp.concatenate([u_a_tangent_a]),
                jnp.concatenate([u_a_tangent_b]),
            ]
        ) * (constraint_type == 0)
        jac_prismatic = jnp.stack(
            [
                jnp.concatenate([dot2_3_a]),
                jnp.concatenate([dot2_3_b]),
            ]
        ) * (constraint_type == 1)

        return jac_hinge + jac_prismatic

    @jit
    def multi_tangential6x1(
        param,
        state,
        body_a_id,
        body_b_id,
        constraint_id,
        constraint_type,
    ):
        body_b_pos = state.conf.pos[body_b_id]
        body_b_rot = state.conf.rot[body_b_id]

        body_a_pos = jnp.array([0.0, 0.0, 0.0])
        body_a_rot = jnp.array([1.0, 0.0, 0.0, 0.0])

        d_a, u_a, v_a, w_a = Constraint.get_frame_transform(
            param.constraint_param[constraint_id].frame_a, body_a_pos, body_a_rot
        )
        d_b, u_b, v_b, w_b = Constraint.get_frame_transform(
            param.constraint_param[constraint_id].frame_b, body_b_pos, body_b_rot
        )

        dot2_3_b = jnp.block([-v_a[None], jnp.zeros([1, 3])])

        u_a_tangent_b = jnp.block([jnp.zeros([1, 3]), -u_a])

        jac_hinge = jnp.stack(
            [
                jnp.concatenate([u_a_tangent_b]),
            ]
        ) * (constraint_type == 0)
        jac_prismatic = jnp.stack(
            [
                jnp.concatenate([dot2_3_b]),
            ]
        ) * (constraint_type == 1)

        return jac_prismatic + jac_hinge
