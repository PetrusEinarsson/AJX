from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct
from typing import Optional, Callable, List, Tuple, LiteralString
from dataclasses import dataclass, field
from jax.tree_util import register_pytree_node_class


@struct.dataclass
class Frame:
    position: jnp.array
    rotation: jnp.array

    def concat(self, axis=-1):
        return jnp.concatenate([self.position, self.rotation], axis=axis)

    def stack(frames):
        pos = jnp.stack([transform.position for transform in frames])
        rot = jnp.stack([transform.rotation for transform in frames])
        return Frame(pos, rot)

    def __getitem__(self, key):
        return Frame(self.position[key], self.rotation[key])


@struct.dataclass
class DryFrictionParameters:
    mu: float
    delta: float
    b: float


@register_pytree_node_class
class ConstraintParameters:
    # Fixed
    names: Tuple[str]

    # Dynamic
    data: jnp.array
    # frame_a: Frame (7 values)
    # frame_b: Frame (7 values)
    # compliance: float (6 values)  (Works as viscous compliance if not holonomic)
    # damping: float (6 values)  (The corresonding dof is nonholonomic if negative)
    # target_offset: (6 values) (Target offset if holonomic, target velocity if nonholonomic)

    def __init__(
        self,
        names,
        data,
    ):
        self.names = names
        self.data = data

    @property
    def frame_a(self):
        position = self.data[..., 0:3]
        rotation = self.data[..., 3:7]
        return Frame(position, rotation)

    @property
    def frame_b(self):
        position = self.data[..., 7:10]
        rotation = self.data[..., 10:14]
        return Frame(position, rotation)

    @property
    def compliance(self):
        return self.data[..., 14:20]

    @property
    def damping(self):
        return self.data[..., 20:26]

    @property
    def damping5(self):
        return self.data[..., 25]

    @property
    def target5(self):
        return self.data[..., 31]

    @property
    def target(self):
        return self.data[..., 26:32]

    @classmethod
    def create(
        cls,
        frame_a: Frame,
        frame_b: Frame,
        compliance: float,
        damping: float,
        b: float,
        name: str,
    ):
        frame_a_data = frame_a.concat()
        frame_b_data = frame_b.concat()
        holonomic_compliance = jnp.array([compliance] * 5)
        damping = jnp.array([damping] * 5)
        viscous_compliance = jnp.array([1.0 / b])
        damping_flag = jnp.array([-1.0])
        target = jnp.zeros(6)
        new_data = jnp.concatenate(
            [
                frame_a_data,
                frame_b_data,
                holonomic_compliance,
                viscous_compliance,
                damping,
                damping_flag,
                target,
            ]
        )
        return cls(name, new_data)

    @classmethod
    def create_locked(
        cls,
        frame_a: Frame,
        frame_b: Frame,
        compliance: float,
        damping: float,
        offset: float,
        name: str,
    ):
        frame_a_data = frame_a.concat()
        frame_b_data = frame_b.concat()
        holonomic_compliance = jnp.array([compliance] * 6)
        damping = jnp.array([damping] * 6)
        target = jnp.zeros(5)
        offset = jnp.array([offset])
        new_data = jnp.concatenate(
            [frame_a_data, frame_b_data, holonomic_compliance, damping, target, offset]
        )
        return cls(name, new_data)

    def stack(param_list: Tuple[ConstraintParameters]):
        data_stacked = jnp.stack([param.data for param in param_list])
        names = tuple([param.names for param in param_list])
        return ConstraintParameters(names, data_stacked)

    def stack_with_constraints(
        param_constraint_pairs: Tuple[Tuple[ConstraintParameters, Constraint]],
    ):
        if not param_constraint_pairs:
            data = jnp.zeros([0])
            return ConstraintParameters(tuple(), data), tuple()
        data_stacked = jnp.stack([pair[0].data for pair in param_constraint_pairs])
        c_names = tuple([pair[0].names for pair in param_constraint_pairs])
        constraints = tuple(pair[1] for pair in param_constraint_pairs)
        c_names2 = tuple(constraint.name for constraint in constraints)
        assert c_names == c_names2
        return ConstraintParameters(c_names, data_stacked), constraints

    def copy(self):
        return ConstraintParameters(self.names, self.data)

    def tree_flatten(self):
        children = (self.data,)
        aux_data = (self.names,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data, *children)

    def __getitem__(self, key):
        if isinstance(key, jax.core.Tracer):
            return ConstraintParameters(None, self.data[key])
        return ConstraintParameters(self.names[key], self.data[key])

    def insert(self, src):
        new = self.copy()
        for constraint_name, src2 in src.items():
            if not constraint_name in self.names:
                msg = f"The provided source ({constraint_name}) does not index destination correctly"
                raise Exception(msg)
            idx = self.names.index(constraint_name)
            if src2 is None:
                continue
            for prop, val in src2.items():
                if prop == "compliance05":
                    new.data = new.data.at[idx, 14:20].set(val)
                elif prop == "compliance04":
                    new.data = new.data.at[idx, 14:19].set(val)
                elif prop == "compliance5":
                    new.data = new.data.at[idx, 19].set(val)
                elif prop == "damping05":
                    new.data = new.data.at[idx, 20:26].set(val)
                elif prop == "damping04":
                    new.data = new.data.at[idx, 20:25].set(val)
                elif prop == "damping5":
                    new.data = new.data.at[idx, 25].set(val)
                elif prop == "frame_b_pos_x":
                    new.data = new.data.at[idx, 7].set(val)
                elif prop == "frame_b_pos_y":
                    new.data = new.data.at[idx, 8].set(val)
                elif prop == "target5":
                    new.data = new.data.at[idx, 31].set(val)
                else:
                    msg = f"The provided source ({prop}) does not index destination correctly"
                    raise Exception(msg)
        return new

    def increment(self, src):
        new = self.copy()
        for constraint_name, src2 in src.items():
            if not constraint_name in self.names:
                msg = f"The provided source ({constraint_name}) does not index destination correctly"
                raise Exception(msg)
            idx = self.names.index(constraint_name)
            if src2 is None:
                continue
            for prop, val in src2.items():
                if prop == "compliance05":
                    eps = self.data[idx, 14:20]
                    n = eps / (1 + val * eps)
                    new.data = new.data.at[idx, 14:20].set(n)
                if prop == "compliance04":
                    eps = self.data[idx, 19]
                    n = eps / (1 + val * eps)
                    new.data = new.data.at[idx, 19].set(n)
                elif prop == "damping04":
                    new.data = new.data.at[idx, 20:26].set(val + self.data[idx, 20:26])
                elif prop == "frame_b_pos_x":
                    new.data = new.data.at[idx, 7].set(val + self.data[idx, 7])
                elif prop == "frame_b_pos_y":
                    new.data = new.data.at[idx, 8].set(val + self.data[idx, 8])
                else:
                    msg = f"The provided source ({prop}) does not index destination correctly"
                    raise Exception(msg)
        return new

    def as_dict(self):
        res = {}
        for i, name in enumerate(self.names):

            res[f"{name}.frame_a_pos_x"] = self.data[i, 0]
            res[f"{name}.frame_a_pos_y"] = self.data[i, 1]
            res[f"{name}.frame_a_pos_z"] = self.data[i, 2]

            res[f"{name}.frame_b_pos_x"] = self.data[i, 7]
            res[f"{name}.frame_b_pos_y"] = self.data[i, 8]
            res[f"{name}.frame_b_pos_z"] = self.data[i, 9]

            res[f"{name}.compliance"] = self.data[i, 14]
            res[f"{name}.damping"] = self.data[i, 20]

        return res


@struct.dataclass
class SensorParameters:
    frame: Frame


@dataclass
class Constraint:
    c: Callable[[List, jnp.array], jnp.array]
    jac: Callable[[List, jnp.array], jnp.array]
    D: Callable[[List, jnp.array], jnp.array]
    D_bar: Callable[[List, jnp.array], jnp.array]
    place_other: Callable[[List, jnp.array, float], jnp.array]
    get_free_degrees: Callable[[jnp.array], jnp.array]
    name: str
    body_a: str
    body_b: str
    dof_removed: str
    is_inequality: bool
    ghosts_endings_normal: List[str] = field(default_factory=lambda: [])
    ghosts_endings_tangential: List[str] = field(default_factory=lambda: [])


def get_ghost_names(constraint: Constraint, param, i: int) -> List[str]:
    """
    Get defualt names for the ghost variables based on the given constraint
    and constraint_param.
    """
    return [f"{constraint.name}.lambda_{s}" for s in constraint.multiplier_suffixes]

    ghost_names = []
    ghosts_normal = constraint.ghosts_endings_normal
    ghosts_tangential = constraint.ghosts_endings_tangential

    a = [f"{constraint.name}.lambda_{ghost}" for ghost in constraint.ghosts_endings]
    ghost_names.extend(a)

    a = [f"{constraint.name}.lambda_{ghost}" for ghost in ghosts_tangential]
    ghost_names.extend(a)

    if "motor" in constraint.blocks:
        a = [f"{constraint.name}.lambda_motor_{ghost}" for ghost in ghosts_tangential]
        ghost_names.extend(a)

    if constraint.enable_dry_friction:
        ghost_names.append(f"{constraint.name}.sigma")
    return ghost_names
