import jax

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)

from ajx.example_graphics.environment_scene import EnvironmentScene
from ajx.example_graphics.application import Application
from ajx.constraints import ConstraintType
from ajx.example_environments.dlo import DLO, DLOSettings
from ajx.simulation import SimulationSettings, Solver
import jax.numpy as jnp

if __name__ == "__main__":
    timestep = 0.016667

    environment = DLO(
        sim_settings=SimulationSettings(timestep, True, Solver.SPARSE_LINEAR),
        env_settings=DLOSettings(
            n_bodies=100,
            body_length=0.04,
            constraint_type=ConstraintType.PRISMATIC.value,
            loose_end=True,
        ),
    )
    yz_compliance = 1e-6
    x_compliance = 1e-2
    bend_compliance = 1e-6
    torsion_compliance = 5e-1

    env_param = environment.default_param.tree_replace(
        src={
            "sparse_param.coupled_constraint_param": {
                "compliance": jnp.array(
                    [
                        x_compliance,
                        yz_compliance,
                        yz_compliance,
                        bend_compliance,
                        bend_compliance,
                        torsion_compliance,
                    ]
                ),
                "is_velocity": jnp.array([0, 0, 0, 0, 0, 0]),
            }
        }
    )

    initial_state = environment.state_from_angles(env_param)
    environment.lock_joints[1].get_free_degrees(initial_state, env_param)

    scene = EnvironmentScene(environment, env_param, initial_state)
    app = Application(scene, 60, "default")
    app.run()
