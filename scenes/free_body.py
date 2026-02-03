from scenes.base import GraphicalEnvironmentBase

from environments.free_body import FreeBody

from ajx import *

if __name__ == "__main__":
    timestep = 0.016667

    environment = FreeBody(
        override_param={},
        timestep=timestep,
        use_gyroscopic=True,
    )
    initial_state = {}
    angvel = jnp.array([0.0, 0.5, 0.5])

    initial_state = environment.state_from_angular_velocity(angvel)

    controller = GraphicalEnvironmentBase(environment, initial_state)
    controller.run()
