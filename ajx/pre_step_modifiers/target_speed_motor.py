from flax import struct
from ajx.pre_step_modifiers.base import PreStepModifier
from ajx.constraints import Constraint
from flax import struct


@struct.dataclass
class TargetSpeedMotor(PreStepModifier):
    name: str
    constraint: Constraint
    idx: int

    def update_params(self, state, u, param):
        target = u[self.idx]
        return state, param.tree_replace(
            {"constraint_param": {"target": {self.constraint.name: {5: target}}}}
        )
