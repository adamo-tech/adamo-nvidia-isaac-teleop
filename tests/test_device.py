"""Round-trip test for the Isaac device-representation codec (device.py).

Fields are packed as float32, so values are chosen to be float32-exact.
"""

from __future__ import annotations

import numpy as np

from adamo_isaac_teleop.device import (
    BUTTONS,
    ControllerDevice,
    decode_controller,
    encode_controller,
)


def test_controller_round_trip():
    dev = ControllerDevice(
        side="right",
        grip_position=np.array([0.5, -1.0, 2.0]),
        grip_orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        aim_position=np.array([0.25, 0.5, -0.5]),
        aim_orientation=np.array([0.5, 0.5, 0.5, 0.5]),
        buttons={"trigger": 1.0, "squeeze": 0.5},
        grip_valid=True,
        aim_valid=False,
    )
    out = decode_controller(encode_controller(dev))

    assert out.side == "right"
    np.testing.assert_allclose(out.grip_position, dev.grip_position)
    np.testing.assert_allclose(out.grip_orientation, dev.grip_orientation)
    np.testing.assert_allclose(out.aim_position, dev.aim_position)
    np.testing.assert_allclose(out.aim_orientation, dev.aim_orientation)
    assert out.grip_valid is True
    assert out.aim_valid is False
    # all buttons present; unset ones default to 0.0
    assert set(out.buttons) == set(BUTTONS)
    assert out.buttons["trigger"] == 1.0
    assert out.buttons["squeeze"] == 0.5
    assert out.buttons["primary"] == 0.0
