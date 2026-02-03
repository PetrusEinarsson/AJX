from __future__ import annotations

from flax import struct
import jax.numpy as jnp
import jax.tree_util
from typing import Tuple, LiteralString
from ajx import math
from jax import vmap
from jax import lax
from jax.tree_util import register_pytree_node_class
import numpy as np


@jax.custom_jvp
def expm(mat):
    return jax.scipy.linalg.expm(mat)


@expm.defjvp
def expm_vjp(primal, tangent):
    mat, tang = primal[0], tangent[0]
    E, E_frechet = jax.scipy.linalg.expm_frechet(mat, tang, compute_expm=True)
    return E, E_frechet


@jax.custom_jvp
def sqrtm_and_inv(mat):
    reg = jnp.eye(mat.shape[0])
    eigvals, eigvecs = jnp.linalg.eigh(mat + reg)
    P_sqrt = eigvecs @ jnp.diag(jnp.sqrt(eigvals)) @ eigvecs.T
    P_sqrt_inv = eigvecs @ jnp.diag(1 / jnp.sqrt(eigvals)) @ eigvecs.T
    return P_sqrt, P_sqrt_inv


@sqrtm_and_inv.defjvp
def sqrtm_and_inv_vjp(primal, tangent):
    mat, tang = primal[0], tangent[0]
    sqrt, sqrt_inv = sqrtm_and_inv(mat)
    sqrt_tangent = jax.scipy.linalg.solve_sylvester(sqrt, sqrt, tang, method="eigen")
    sqrt_inv_tangent = -sqrt_inv @ sqrt_tangent @ sqrt_inv
    return (sqrt, sqrt_inv), (sqrt_tangent, sqrt_inv_tangent)


@register_pytree_node_class
class RigidBodyParameters:
    # Fixed
    names: Tuple[str]

    # Dynamic
    data: jnp.array

    def __init__(
        self,
        names,
        data,
    ):
        self.names = names
        self.data = data

    @property
    def mass(self):
        return self.data[..., 0]

    @property
    def mc(self):
        return self.data[..., 1:4]

    @property
    def inertia(self):
        inertia = jnp.zeros([3, 3])
        if len(self.data.shape) == 2:
            n_bodies = self.data.shape[0]
            inertia = jnp.zeros([n_bodies, 3, 3])
        triu = np.triu_indices(3)
        tril = (triu[1], triu[0])
        if len(self.data.shape) == 2:
            inertia = vmap(lambda x, y: x.at[triu].set(y))(
                inertia, self.data[..., 4:10]
            )
            inertia = vmap(lambda x, y: x.at[triu].set(y))(
                inertia, self.data[..., 4:10]
            )
        else:
            inertia = inertia.at[triu].set(self.data[4:10])
            inertia = inertia.at[tril].set(self.data[4:10])

        return inertia

    def pseudo_inertia(data):
        m = data[0]
        mass_block = jnp.diag(jnp.array([m]))
        mc = data[1:4, None]
        inertia = jnp.zeros([3, 3])
        triu = np.triu_indices(3)
        tril = (triu[1], triu[0])
        inertia = inertia.at[triu].set(data[4:10])
        inertia = inertia.at[tril].set(data[4:10])
        return jnp.block([[inertia, mc], [mc.T, mass_block]])

    @classmethod
    def create(cls, mass: float, inertia_diag: jax.Array, name: str):
        mass = jnp.array(mass).reshape(1)
        mc = jnp.array([0.0, 0.0, 0.0])
        diag_indices = jnp.array([0, 3, 5])
        inertia = jnp.zeros(6).at[diag_indices].set(inertia_diag)

        new_data = jnp.concatenate([mass, mc, inertia])
        return cls(name, new_data)

    def stack(param_list: Tuple[RigidBodyParameters]):
        data_stacked = jnp.stack([param.data for param in param_list])
        names = tuple([param.names for param in param_list])
        return RigidBodyParameters(names, data_stacked)

    def stack_with_rigid_bodies(
        param_rb_pairs: Tuple[Tuple[RigidBodyParameters, RigidBody]],
    ):
        data_stacked = jnp.stack([param_rb[0].data for param_rb in param_rb_pairs])
        rb_names = tuple([param_rb[0].names for param_rb in param_rb_pairs])
        rigid_bodies = tuple(param_rb[1] for param_rb in param_rb_pairs)
        rb_names2 = tuple(rb.name for rb in rigid_bodies)
        assert rb_names == rb_names2
        return RigidBodyParameters(rb_names, data_stacked), rigid_bodies

    def copy(self):
        return RigidBodyParameters(self.names, self.data)

    def tree_flatten(self):
        children = (self.data,)
        aux_data = (self.names,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data, *children)

    def __getitem__(self, key):
        return RigidBodyParameters(self.names[key], self.data[key])

    def insert(self, src):
        new = self.copy()
        if src is None:
            return new
        if isinstance(src, jax.Array):
            # TODO: Lazy fix for empty dict
            return new
        for rb_name, src2 in src.items():
            if not rb_name in self.names:
                msg = f"The provided source ({rb_name}) does not index destination correctly"
                raise Exception(msg)
            idx = self.names.index(rb_name)
            for prop, val in src2.items():
                if prop == "mass":
                    new.data = new.data.at[idx, 0].set(val)
                elif prop == "mc_x":
                    new.data = new.data.at[idx, 1].set(val)
                elif prop == "mc_y":
                    new.data = new.data.at[idx, 2].set(val)
                elif prop == "mc_z":
                    new.data = new.data.at[idx, 3].set(val)
                elif prop == "inertia_xx":
                    new.data = new.data.at[idx, 4].set(val)
                elif prop == "inertia_xy":
                    new.data = new.data.at[idx, 5].set(val)
                elif prop == "inertia_xz":
                    new.data = new.data.at[idx, 6].set(val)
                elif prop == "inertia_yy":
                    new.data = new.data.at[idx, 7].set(val)
                elif prop == "inertia_yz":
                    new.data = new.data.at[idx, 8].set(val)
                elif prop == "inertia_zz":
                    new.data = new.data.at[idx, 9].set(val)
                else:
                    msg = f"The provided source ({prop}) does not index destination correctly"
                    raise Exception(msg)
        return new

    def increment(self, src, update_type="euclidian"):
        new = self.copy()
        if src is None:
            return new
        if isinstance(src, jax.Array):
            # TODO: Lazy fix for empty dict
            return new
        for rb_name, src2 in src.items():
            if not rb_name in self.names:
                msg = f"The provided source ({rb_name}) does not index destination correctly"
                raise Exception(msg)
            idx = self.names.index(rb_name)
            increment = jnp.zeros_like(new.data[idx])
            for prop, val in src2.items():
                if prop == "mass":
                    increment = increment.at[0].set(val)
                elif prop == "mc_x":
                    increment = increment.at[1].set(val)
                elif prop == "mc_y":
                    increment = increment.at[2].set(val)
                elif prop == "mc_z":
                    increment = increment.at[3].set(val)
                elif prop == "inertia_xx":
                    increment = increment.at[4].set(val)
                elif prop == "inertia_xy":
                    increment = increment.at[5].set(val)
                elif prop == "inertia_xz":
                    increment = increment.at[6].set(val)
                elif prop == "inertia_yy":
                    increment = increment.at[7].set(val)
                elif prop == "inertia_yz":
                    increment = increment.at[8].set(val)
                elif prop == "inertia_zz":
                    increment = increment.at[9].set(val)
                else:
                    msg = f"The provided source ({prop}) does not index destination correctly"
                    raise Exception(msg)

            if update_type == "euclidian":
                new.data = new.data.at[idx].set(new.data[idx] + increment)
            elif update_type == "inverse_euclidian":
                inv_incremented = new.data[idx] / (1 + increment * new.data[idx])
                new.data = new.data.at[idx].set(inv_incremented)
            elif update_type == "inverse_project":
                inv_incremented = new.data[idx] / (1 + increment * new.data[idx])
                new.data = jnp.clip(new.data.at[idx].set(inv_incremented), min=1e-4)
            elif update_type == "project":
                new.data = jnp.clip(
                    new.data.at[idx].set(new.data[idx] + increment), min=1e-4
                )
            elif update_type == "lazy_airm":
                P = RigidBodyParameters.pseudo_inertia(new.data[idx])
                X = RigidBodyParameters.pseudo_inertia(increment)
                p = jnp.diag(P)
                x = jnp.diag(X)
                res_vec = p * jnp.exp(x / (p + 1e-16))
                res = jnp.diag(res_vec)
                mass = res[3, 3]
                mc = res[0:3, 3]
                inertia_xx = res[0, 0]
                inertia_xy = res[0, 1]
                inertia_xz = res[0, 2]
                inertia_yy = res[1, 1]
                inertia_yz = res[1, 2]
                inertia_zz = res[2, 2]
                res_vec = jnp.stack(
                    [
                        mass,
                        mc[0],
                        mc[1],
                        mc[2],
                        inertia_xx,
                        inertia_xy,
                        inertia_xz,
                        inertia_yy,
                        inertia_yz,
                        inertia_zz,
                    ]
                )

                # def is_spos_def(x):
                #     return jnp.all(jnp.linalg.eigvals(x) >= 0)

                # def is_pos_def(x):
                #     return jnp.all(jnp.linalg.eigvals(x) >= 0)

                new.data = new.data.at[idx].set(res_vec)
            elif update_type == "airm":
                # Affine Invariant Riemannian Metric
                P = RigidBodyParameters.pseudo_inertia(new.data[idx])
                X = RigidBodyParameters.pseudo_inertia(increment)

                P_half, P_inv_half = sqrtm_and_inv(P)
                exp_part = expm(P_inv_half @ X @ P_inv_half)
                res = P_half @ exp_part @ P_half
                res = 0.5 * (res + res.T)

                # P_lbda, P_v = jnp.linalg.eigh(P)
                # P_half = P_v @ jnp.diag(jnp.sqrt(P_lbda)) @ P_v.T
                # P_half_inv = P_v @ jnp.diag(1 / jnp.sqrt(P_lbda)) @ P_v.T
                # A = P_half_inv @ X @ P_half_inv
                # A = jnp.linalg.solve(P, X)
                # A_lbda, A_v = expm(A)
                # exp_A = A_v @ jnp.diag(jnp.exp(A_lbda)) @ A_v.T
                # exp_A = expm(A)
                # res = P_half @ exp_A @ P_half
                # res = P @ exp_A
                # res = res @ P @ res
                mass = res[3, 3]
                mc = res[0:3, 3]
                inertia_xx = res[0, 0]
                inertia_xy = res[0, 1]
                inertia_xz = res[0, 2]
                inertia_yy = res[1, 1]
                inertia_yz = res[1, 2]
                inertia_zz = res[2, 2]
                res_vec = jnp.stack(
                    [
                        mass,
                        mc[0],
                        mc[1],
                        mc[2],
                        inertia_xx,
                        inertia_xy,
                        inertia_xz,
                        inertia_yy,
                        inertia_yz,
                        inertia_zz,
                    ]
                )

                # def is_spos_def(x):
                #     return jnp.all(jnp.linalg.eigvals(x) >= 0)

                # def is_pos_def(x):
                #     return jnp.all(jnp.linalg.eigvals(x) >= 0)

                new.data = new.data.at[idx].set(res_vec)
            else:
                raise NotImplementedError
        return new

    def as_dict(self):
        res = {}
        for i, name in enumerate(self.names):
            res[f"{name}.mass"] = self.data[i, 0]
            res[f"{name}.mc_x"] = self.data[i, 1]
            res[f"{name}.mc_y"] = self.data[i, 2]
            res[f"{name}.mc_z"] = self.data[i, 3]
            res[f"{name}.inertia_xx"] = self.data[i, 4]
            res[f"{name}.inertia_xy"] = self.data[i, 5]
            res[f"{name}.inertia_xz"] = self.data[i, 6]
            res[f"{name}.inertia_yy"] = self.data[i, 7]
            res[f"{name}.inertia_yz"] = self.data[i, 8]
            res[f"{name}.inertia_zz"] = self.data[i, 9]
        return res


@struct.dataclass
class RigidBody:
    name: str
    geometry: Tuple[str]  # For 3D graphics (not handled by the physics engine)


@struct.dataclass
class Configuration:
    pos: jax.Array
    rot: jax.Array

    def concat(self, axis=-1):
        return jnp.concatenate([self.pos, self.rot], axis=axis)

    def stack(transforms):
        pos = jnp.stack([transform.pos for transform in transforms])
        rot = jnp.stack([transform.rot for transform in transforms])
        return Configuration(pos, rot)

    def __getitem__(self, key):
        return Configuration(self.pos[key], self.rot[key])

    @property
    def shape(self):
        return tuple(map(sum, zip(self.pos.shape, self.rot.shape)))


@struct.dataclass
class GeneralizedVelocity:
    data: jax.Array

    @property
    def vel(self):
        return self.data[..., :3]

    @property
    def ang(self):
        return self.data[..., 3:]

    def stack(transforms):
        data = jnp.stack([transform.data for transform in transforms])
        return GeneralizedVelocity(data)

    def __getitem__(self, key):
        return GeneralizedVelocity(self.data[key])

    @property
    def shape(self):
        return self.data.shape


@struct.dataclass
class State:
    conf: Configuration
    gvel: GeneralizedVelocity

    def __getitem__(self, key):
        return State(self.conf[key], self.gvel[key])

    def pack(self):
        return jnp.concatenate(
            [self.conf.pos, self.conf.rot, self.gvel.data], axis=None
        )

    def stack(transforms):
        conf = Configuration.stack([transform.conf for transform in transforms])
        gvel = GeneralizedVelocity.stack([transform.gvel for transform in transforms])
        return State(conf, gvel)

    def pack_w_zero_rot(self, packed_format: str):
        n_timesteps = self.conf.pos.shape[0]
        if packed_format == "pos-rot-gvel":
            pos = self.conf.pos.reshape(n_timesteps, -1)
            delta_rot = jnp.zeros_like(pos)
            gvel = self.gvel.data.reshape(n_timesteps, -1)
            return jnp.concatenate([pos, delta_rot, gvel], axis=1)
        if packed_format == "conf-gvel":
            pos = self.conf.pos
            delta_rot = jnp.zeros_like(pos)
            gvel = self.gvel.data.reshape(n_timesteps, -1)
            conf = jnp.concatenate([pos, delta_rot], axis=2).reshape(n_timesteps, -1)
            return jnp.concatenate([conf, gvel], axis=1)
        raise NotImplementedError

    def pack_quaternion_rotation(self):
        return self.conf.rot

    def increment(self, delta, format):
        if len(self.gvel.shape) == 2:
            n_bodies = self.gvel.shape[0]
            if format == "conf-gvel":
                # TODO: May not be compatible with custom jacobian
                delta_conf = delta[..., : 6 * n_bodies].reshape(n_bodies, 6)
                delta_gvel = delta[..., 6 * n_bodies :].reshape(n_bodies, 6)
                delta_pos = delta_conf[..., :3]
                delta_rot = delta_conf[..., 3:]

                new_pos = self.conf.pos + delta_pos
                new_gvel = self.gvel.data + delta_gvel
                qdelta_rot = vmap(math.from_rotation_vector)(delta_rot)
                new_rot = vmap(math.quat_mul)(qdelta_rot, self.conf.rot)

                return State(
                    Configuration(new_pos, new_rot), GeneralizedVelocity(new_gvel)
                )
        assert len(self.gvel.shape) == 3
        n_timesteps = self.gvel.shape[0]
        n_bodies = self.gvel.shape[1]
        delta = delta.reshape(n_timesteps, n_bodies * 12)

        if format == "conf-gvel":
            # TODO: May not be compatible with custom jacobian
            delta_conf = delta[..., : 6 * n_bodies].reshape(n_timesteps, n_bodies, 6)
            delta_gvel = delta[..., 6 * n_bodies :].reshape(n_timesteps, n_bodies, 6)
            delta_pos = delta_conf[..., :3]
            delta_rot = delta_conf[..., 3:]

            new_pos = self.conf.pos + delta_pos
            new_gvel = self.gvel.data + delta_gvel
            qdelta_rot = vmap(vmap(math.from_rotation_vector))(delta_rot)
            new_rot = vmap(vmap(math.quat_mul))(qdelta_rot, self.conf.rot)

            return State(Configuration(new_pos, new_rot), GeneralizedVelocity(new_gvel))
        raise NotImplementedError


@struct.dataclass
class Trajectory2:
    data: jax.Array

    def pack_w_delta_rot_representation(self):
        assert self.data.shape[1] % 13 == 0
        n_bodies = self.data.shape[1] // 13
        pos = self.data[:, : 3 * n_bodies]
        delta_rot = jnp.zeros_like(pos)
        gvel = self.data[:, 7 * n_bodies :]
        return jnp.concatenate([pos, delta_rot, gvel], axis=1)

    def pack_quaternion_rotation(self):
        assert self.data.shape[1] % 13 == 0
        n_bodies = self.data.shape[1] // 13
        rot = self.data[:, 3 * n_bodies : 7 * n_bodies]
        return rot

    def unpack_w_delta_rot(arr, qrot_ref, format):
        assert arr.shape[1] % 12 == 0
        n_bodies = arr.shape[1] // 12
        n_timesteps = arr.shape[0]
        if format == "pos-rot-gvel":
            pos = arr[:, : 3 * n_bodies].reshape(-1, n_bodies, 3)
            delta_rot = arr[:, 3 * n_bodies : 6 * n_bodies].reshape(-1, n_bodies, 3)
            gvel = arr[:, 6 * n_bodies :].reshape(-1, n_bodies, 6)

            qdelta_rot = vmap(vmap(math.from_rotation_vector))(delta_rot)
            qrot_ref = qrot_ref.reshape(n_timesteps, n_bodies, 4)
            rot = vmap(vmap(math.quat_mul))(qdelta_rot, qrot_ref)

            return State(Configuration(pos, rot), GeneralizedVelocity(gvel))
        if format == "conf-gvel":
            conf = arr[:, : 6 * n_bodies].reshape(-1, n_bodies, 6)
            gvel = arr[:, 6 * n_bodies :].reshape(-1, n_bodies, 6)
            pos = conf[..., :3]
            delta_rot = conf[..., 3:]

            qdelta_rot = vmap(vmap(math.from_rotation_vector))(delta_rot)
            qrot_ref = qrot_ref.reshape(n_timesteps, n_bodies, 4)
            rot = vmap(vmap(math.quat_mul))(qdelta_rot, qrot_ref)

            return State(Configuration(pos, rot), GeneralizedVelocity(gvel))
        raise NotImplementedError

    @property
    def shape(self):
        return self.data.shape


if __name__ == "__main__":
    pass
    x = jnp.linspace(0, 1, 101)
    y = x + 1
    z = x + 2
    pos = jnp.stack([x, y, z], axis=-1)
    rot = jnp.array([1.0, 0.0, 0.0, 0.0])
    rot = rot[None].repeat(101, axis=0)
    gvel = jnp.stack([x + 3, x + 4, x + 5, x + 6, x + 7, x + 8], axis=-1)

    data = jnp.concatenate([pos, rot, gvel], axis=1)
    traj = Trajectory(data)
    packed = traj.pack_w_delta_rot_representation()
    quat = traj.pack_quaternion_rotation()

    reconstructed = Trajectory.unpack_w_delta_rot_representation(packed, quat)
    reconstructed_pos = reconstructed.conf.pos[:, 0]
    reconstructed_rot = reconstructed.conf.rot[:, 0]
    reconstructed_gvel = reconstructed.gvel.data[:, 0]

    assert jnp.all(reconstructed_pos == pos)
    assert jnp.all(reconstructed_rot == rot)
    assert jnp.all(reconstructed_gvel == gvel)

    packed = reconstructed.pack_w_delta_rot_representation()
    quat = reconstructed.pack_quaternion_rotation()

    reconstructed = Trajectory.unpack_w_delta_rot_representation(packed, quat)
    reconstructed_pos = reconstructed.conf.pos[:, 0]
    reconstructed_rot = reconstructed.conf.rot[:, 0]
    reconstructed_gvel = reconstructed.gvel.data[:, 0]

    assert jnp.all(reconstructed_pos == pos)
    assert jnp.all(reconstructed_rot == rot)
    assert jnp.all(reconstructed_gvel == gvel)

    pass
