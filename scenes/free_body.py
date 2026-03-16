from ajx.example_graphics.environment_scene import EnvironmentScene
from ajx.example_graphics.application import Application

from ajx.example_environments.free_body import FreeBody
from ajx.simulation import SimulationSettings, Solver
import jax.numpy as jnp

if __name__ == "__main__":
    timestep = 0.016667

    environment = FreeBody(
        sim_settings=SimulationSettings(timestep, True, Solver.DENSE_LINEAR)
    )
    env_param = environment.default_param.tree_replace(src={})
    angvel = jnp.array([0.0, 0.5, 0.5])

    initial_state = environment.state_from_angular_velocity(angvel)

    scene = EnvironmentScene(environment, env_param, initial_state)
    app = Application(scene, 60, "default")
    app.run()
