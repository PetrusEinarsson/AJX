from scenes.base import GraphicalEnvironmentBase

from environments.cartpole import CartPole
from functools import partial

from ajx import *

if __name__ == "__main__":
    timestep = 0.016667

    environment = CartPole(
        override_param={},
        timestep=timestep,
    )

    initial_state = environment.state_from_angles(5.0, 3.0, environment.param)

    controller = GraphicalEnvironmentBase(environment, initial_state)
    controller.run()
