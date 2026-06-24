"""Test bootstrap.

Importing ``adamo_isaac_teleop`` pulls in ``retarget`` -> ``isaacteleop``. The codec
tests (cdr/device) don't touch that dependency, so when it isn't installed we stub it
just enough for the package to import. With isaacteleop present, the real module is used.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

try:  # use the real dependency when it's available
    import isaacteleop  # noqa: F401
except ImportError:
    _MODULES = [
        "isaacteleop",
        "isaacteleop.retargeting_engine",
        "isaacteleop.retargeting_engine.interface",
        "isaacteleop.retargeting_engine.tensor_types",
        "isaacteleop.retargeters",
        "isaacteleop.retargeters.se3_retargeter",
        "isaacteleop.retargeters.gripper_retargeter",
    ]
    for _name in _MODULES:
        _mod = types.ModuleType(_name)
        _mod.__getattr__ = lambda _attr: MagicMock()  # type: ignore[attr-defined]
        sys.modules[_name] = _mod
    sys.modules["isaacteleop.retargeting_engine.tensor_types"].NUM_HAND_JOINTS = 26
