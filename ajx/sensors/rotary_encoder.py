import jax.numpy as jnp
from ajx.sensors.base import Sensor


class RotaryEncoderHingeMounted(Sensor):
    def __init__(self, name: str, hinge):
        self.name = name
        self.hinge = hinge

        self.observable_names = ["theta"]
        self.residual_names = ["theta"]

    def observe(self, state, qdot_next, param):
        theta1 = self.hinge.get_free_degrees(state, param)
        return jnp.stack(
            [
                theta1,
            ]
        )

    def residual(self, target, prediction):
        angles_prediction = target
        angles_target = prediction
        delta = angles_prediction - angles_target
        cyclic_delta = (delta + jnp.pi) % (2 * jnp.pi) - jnp.pi
        return jnp.concatenate([cyclic_delta])
