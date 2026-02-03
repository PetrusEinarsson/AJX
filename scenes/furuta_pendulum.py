from scenes.base import GraphicalEnvironmentBase

from environments.furuta import Furuta
from functools import partial

from ajx import *


if __name__ == "__main__":
    timestep = 0.016667

    environment = Furuta(
        override_param={},
        timestep=timestep,
        reference_timestep=timestep,
        use_gyroscopic=True,
    )
    initial_state = {}
    theta1 = 1.0
    theta2 = 4.0

    initial_state = environment.state_from_angles(theta1, theta2, environment.param)
    controller = GraphicalEnvironmentBase(environment, initial_state)
    controller.run()
