from __future__ import annotations

import sys
import jax

from ajx.definitions import (
    ConstraintParameters,
    RigidBodyParameters,
    ScalarBodyParameters,
    ScalarConstraintParameters,
)


from ajx.tree_util import ParameterNode
from flax import struct
from typing import Tuple


@struct.dataclass
class SimulationParameters(ParameterNode):
    # Dynamic
    gravity: jax.Array
    rigid_body_param: RigidBodyParameters
    constraint_param: ConstraintParameters
    sparse_param: ParameterNode
    scalar_body_param: ScalarBodyParameters = struct.field(
        default_factory=lambda: ScalarBodyParameters.create_empty()
    )
    scalar_constraint_param: ScalarConstraintParameters = struct.field(
        default_factory=lambda: ScalarConstraintParameters.create_empty()
    )

    # Static
    tangent_restrictions: Tuple[str] = struct.field(pytree_node=False, default=tuple())


def create_parameter_node(name: str, keys: Tuple[str]):
    namespace = {
        "__module__": __name__,
        "__annotations__": {k: object for k in keys},
    }
    cls = type(name, (ParameterNode,), namespace)

    setattr(sys.modules[__name__], name, cls)
    return struct.dataclass(cls)
