import jax.numpy as jnp
from ajx.sensors.base import Sensor
from ajx.definitions import ParameterNode
from flax import struct
import jax
from typing import Tuple


@struct.dataclass
class OffsetParameters(ParameterNode):
    names: Tuple[str] = struct.field(pytree_node=False)
    offset: jax.Array
    scale: jax.Array


class RotaryEncoder(Sensor):
    def __init__(self, name: str, hinge):
        self.name = name
        self.hinge = hinge

        self.observable_names = ["theta"]
        self.residual_names = ["theta"]

    def observe(self, state, qdot_next, param):
        theta1 = self.hinge.get_free_degrees(state, param)
        idx = param.sparse_param.offset_param.names.index(self.name)
        offset = param.sparse_param.offset_param.offset[idx]
        scale = param.sparse_param.offset_param.scale[idx]
        return jnp.stack(
            [
                scale * theta1 + offset,
            ]
        )

    def residual(self, target, prediction):
        angles_prediction = target
        angles_target = prediction
        delta = angles_prediction - angles_target
        cyclic_delta = (delta + jnp.pi) % (2 * jnp.pi) - jnp.pi
        return jnp.concatenate([cyclic_delta])


class LinearEncoder(Sensor):
    def __init__(self, name: str, prismatic):
        self.name = name
        self.prismatic = prismatic

        self.observable_names = ["x"]
        self.residual_names = ["x"]

    def observe(self, state, qdot_next, param):
        theta1 = self.prismatic.get_free_degrees(state, param)
        idx = param.sparse_param.offset_param.names.index(self.name)
        offset = param.sparse_param.offset_param.offset[idx]
        return jnp.stack(
            [
                theta1 + offset,
            ]
        )

    def residual(self, target, prediction):
        delta = prediction - target
        return jnp.concatenate([delta])
