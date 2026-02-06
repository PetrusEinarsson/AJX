import jax.numpy as jnp
from ajx.sensors.base import Sensor


class PrismaticEncoder(Sensor):
    def __init__(self, name: str, prismatic):
        self.name = name
        self.prismatic = prismatic

        self.observable_names = ["x"]
        self.residual_names = ["x"]

    def observe(self, state, qdot_next, param):
        x = self.prismatic.get_free_degrees(state, param)
        return jnp.stack([x])

    def residual(self, target, prediction):
        return target - prediction
