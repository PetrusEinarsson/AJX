import jax.numpy as jnp
from ajx.sensors.base import Sensor
from ajx.definitions import ParameterNode, Transform
from flax import struct
import jax
from typing import Tuple, List
import ajx.math as math
from ajx.param import SimulationParameters
from jax import vmap


@struct.dataclass
class OffsetParameters(ParameterNode):
    names: Tuple[str] = struct.field(pytree_node=False)
    offset: jax.Array
    scale: jax.Array


def get_pos_camera_frame(camera_transform: Transform, pos):
    pos_c = pos - camera_transform.pos
    pos_c2 = math.rotate_vector(math.conjugate(camera_transform.rot), pos_c)
    return pos_c2


def to_global_coords(rb_transform: Transform, pos):
    d = math.rotate_vector(rb_transform.rot, pos)
    return d + rb_transform.pos


class PointTrackingCamera(Sensor):
    def __init__(
        self,
        name: str,
        points: List[Tuple[str, jax.Array]],
        camera_transform: Transform,
    ):
        self.name = name
        # Projection matrix
        self.camera_transform = camera_transform
        self.focal_width = 1.0
        self.focal_height = 1.0

        # Points Tuple[Offset, body]
        self.points = points

        self.observable_names = [
            f"{point[0]}.{e}" for point in points for e in ["x", "y"]
        ]
        self.residual_names = [
            f"{point[0]}.{e}" for point in points for e in ["x", "y"]
        ]
        self.indices = jnp.array([point[0] for point in points], dtype=int)
        self.positions = jnp.stack([point[1] for point in points])

    def observe(self, state, qdot_next, param: SimulationParameters):
        # TODO: Vectorize
        rb_positions = state.conf.pos[self.indices]
        rb_rotations = state.conf.rot[self.indices]

        point_pos = vmap(to_global_coords)(
            Transform(rb_positions, rb_rotations), self.positions
        )
        point_rel_camera = vmap(get_pos_camera_frame, in_axes=(None, 0))(
            self.camera_transform, point_pos
        )
        projected_x = point_rel_camera[:, 0] / point_rel_camera[:, 2] * self.focal_width
        projected_y = (
            point_rel_camera[:, 1] / point_rel_camera[:, 2] * self.focal_height
        )

        return jnp.stack([projected_x, projected_y], axis=1).flatten()

    def residual(self, target, prediction):
        angles_prediction = target
        angles_target = prediction
        delta = angles_prediction - angles_target
        cyclic_delta = (delta + jnp.pi) % (2 * jnp.pi) - jnp.pi
        return jnp.concatenate([cyclic_delta])
