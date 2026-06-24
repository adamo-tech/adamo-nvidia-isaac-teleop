"""Tests for the device-seam codec (device.py).

The wire layer (`_pack`/`_unpack`) is pure numpy/struct and runs everywhere. The
ControllerInput round-trip needs the real ``isaacteleop`` and is skipped when it isn't
installed (the conftest stub can't reconstruct a real tensor group).
"""

from __future__ import annotations

import numpy as np
import pytest

from adamo_isaac_teleop.device import BUTTONS, _pack, _unpack


def test_wire_round_trip():
    # float32-exact values so the round trip is exact
    side = "right"
    grip_p = [0.5, -1.0, 2.0]
    grip_q = [0.0, 0.0, 0.0, 1.0]
    aim_p = [0.25, 0.5, -0.5]
    aim_q = [0.5, 0.5, 0.5, 0.5]
    button_vals = [0.0, 1.0, 0.0, 0.25, -0.5, 0.0, 1.0]  # len == len(BUTTONS)

    s, gp, gq, ap, aq, bv, gv, av = _unpack(
        _pack(side, grip_p, grip_q, aim_p, aim_q, button_vals, True, False))

    assert s == "right"
    np.testing.assert_allclose(gp, grip_p)
    np.testing.assert_allclose(gq, grip_q)
    np.testing.assert_allclose(ap, aim_p)
    np.testing.assert_allclose(aq, aim_q)
    np.testing.assert_allclose(bv, button_vals, atol=1e-7)
    assert gv is True and av is False
    assert len(button_vals) == len(BUTTONS)


def test_controller_input_round_trip():
    """End-to-end through Isaac's ControllerInput; needs the real library."""
    import isaacteleop
    if getattr(isaacteleop, "__is_stub__", False):
        pytest.skip("isaacteleop is stubbed; real library required")
    from isaacteleop.retargeting_engine.tensor_types import ControllerInputIndex

    from adamo_isaac_teleop.device import (
        decode_controller,
        encode_controller,
        make_controller_input,
        trigger_value,
    )

    ci = make_controller_input(
        grip_position=[0.5, -1.0, 2.0],
        grip_orientation=[0.0, 0.0, 0.0, 1.0],
        buttons={"trigger": 1.0, "squeeze": 0.5},
    )
    side, out = decode_controller(encode_controller("left", ci))

    assert side == "left"
    np.testing.assert_allclose(
        np.from_dlpack(out[ControllerInputIndex.GRIP_POSITION]), [0.5, -1.0, 2.0])
    assert trigger_value(out) == 1.0
    assert _scalar_squeeze(out, ControllerInputIndex) == 0.5


def _scalar_squeeze(ci, idx):
    return float(np.reshape(np.from_dlpack(ci[idx.SQUEEZE_VALUE]), -1)[0])
