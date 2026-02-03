from flax import struct
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Literal
from ajx.pre_step_modifiers.base import PreStepModifier


@struct.dataclass
class AxisSymmetricInertiaParam:
    x: float
    y: float


class AxisSymmetricInertia(PreStepModifier):
    def __init__(
        self,
        name: str,
        target_body: str,
        invert: bool,
        order: Literal["yxx", "xyx", "xxy"],
    ):
        self.name = name
        self.target_body = target_body
        self.invert = invert
        self.order = order

    def update_params(self, state, u, param):
        as_inertia_param_x = param[self.name].x
        as_inertia_param_y = param[self.name].y
        if self.invert:
            as_inertia_param_x = 1 / as_inertia_param_x
            as_inertia_param_y = 1 / as_inertia_param_y

        if self.order == "yxx":
            return {
                self.target_body: {
                    "inertia": {
                        0: as_inertia_param_y,
                        1: as_inertia_param_x,
                        2: as_inertia_param_x,
                    }
                }
            }
        elif self.order == "xyx":
            return {
                self.target_body: {
                    "inertia": {
                        0: as_inertia_param_x,
                        1: as_inertia_param_y,
                        2: as_inertia_param_x,
                    }
                }
            }
        return {
            self.target_body: {
                "inertia": {
                    0: as_inertia_param_x,
                    1: as_inertia_param_x,
                    2: as_inertia_param_y,
                }
            }
        }
