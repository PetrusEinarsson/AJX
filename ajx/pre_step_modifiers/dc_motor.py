from flax import struct
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Literal
from ajx.pre_step_modifiers.base import PreStepModifier


@struct.dataclass
class DCMotorParameters:
    # Gamma nu =  G vk + gamma
    # [speed_constant] = [1/emf_constant] = (rad/s)/V
    # target_speed = gamma = -U(t)/emf_constant
    # regularization = Gamma = R/(torque_constant * emf_constant)
    # = R/toruq_constant * speed_constant

    # For ..
    # speed_constant = 116 rpm/V = 14.6 (rad/s) / V
    # torque_constant = 82.2 mNm/A =0.082 Nm/A
    # Resistance = 2.44 Ohm
    # regularization = 434 = 2.44 / 0.082 * 14.6
    # b = 230e-5 Nm / (rad/s)
    # DCMotorParameters(14.6, 230e-5)
    speed_constant: float
    b: float


class DCMotor(PreStepModifier):
    def __init__(self, name: str, target_constraint: str):
        self.name = name
        self.target_constraint = target_constraint

    def update_params(self, state, u, param):
        return {
            self.target_constraint: {
                "motor": {
                    "speed": -u[0] * param[self.name].speed_constant,
                    "b": param[self.name].b,
                }
            }
        }
