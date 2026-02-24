from flax import struct
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Literal
from ajx.pre_step_modifiers.base import PreStepModifier
from ajx.constraints import Constraint
from flax import struct
import jax.numpy as jnp
import jax


@struct.dataclass
class TargetSpeedMotor(PreStepModifier):
    name: str
    constraint: Constraint
    idx: int

    def update_params(self, state, u, param):
        target = u[self.idx]
        return state, param.tree_replace(
            {
                "constraint_param": {
                    self.constraint.name: {
                        "target.5": target,
                    }
                }
            }
        )
