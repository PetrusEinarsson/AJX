from flax import struct
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Literal
from ajx.pre_step_modifiers.base import PreStepModifier


@struct.dataclass
class GainMotorParameters:
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
    inertia: float
    gain: float


class GainMotor(PreStepModifier):
    def __init__(self, name: str, target_constraint, target_constraint_name, timestep):
        self.name = name
        self.target_constraint = target_constraint
        self.target_constraint_name = target_constraint_name
        self.timestep = timestep

    def update_params(self, state, u, param):
        jac = self.target_constraint.tangential_projection(param, state)
        omega = sum([val @ state.gvel.data[body] for body, val in jac.items()])

        speed = (
            omega[0]
            - self.timestep**2
            / param.sparse_param[self.name].inertia
            * param.sparse_param[self.name].gain
            * u[0]
        )
        return {
            "motor_param": {
                self.target_constraint_name: {
                    "speed": speed,
                    "b": param.sparse_param[self.name].inertia / self.timestep,
                }
            }
        }


class GainMotor2(PreStepModifier):
    def __init__(self, name: str, constraint, timestep: float, idx: int):
        self.name = name
        self.constraint = constraint
        self.timestep = timestep
        self.idx = idx

    def update_params(self, state, u, param):
        jac = self.constraint.tangential_projection(param, state)
        omega = sum([val @ state.gvel.data[body] for body, val in jac.items()])

        speed = (
            omega[0]
            - self.timestep**2
            / param.sparse_param[self.name].inertia
            * param.sparse_param[self.name].gain
            * u[self.idx]
        )
        return {
            "constraint_param": {
                self.constraint.name: {
                    "target5": speed,
                    "compliance5": self.timestep
                    / param.sparse_param[self.name].inertia,
                }
            }
        }
