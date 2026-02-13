from __future__ import annotations

from typing import Dict

import jax


from ajx.definitions import ConstraintParameters, RigidBodyParameters


from ajx.tree_util import ParameterNode
from flax import struct


@struct.dataclass
class SimulationParameters(ParameterNode):
    # Dynamic
    gravity: jax.Array
    rigid_body_param: RigidBodyParameters
    constraint_param: ConstraintParameters
    sparse_param: Dict
