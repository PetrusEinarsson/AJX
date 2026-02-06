import jax.numpy as jnp
import ajx.math as math
from ajx.sensors.base import Sensor


class AbsoluteRotationEncoder(Sensor):
    def __init__(self, name: str, body: str):
        self.name = name
        self.body = body

        self.observable_names = ["qs", "qx", "qy", "qz"]
        self.residual_names = ["rx", "ry", "rz"]

    def observe(self, state, qdot_next, param):
        idx = param.rigid_body_param.names.index(self.body)
        return state.conf.rot[idx]

    def residual(self, target, prediction):
        return math.quat_residual(target, prediction)
