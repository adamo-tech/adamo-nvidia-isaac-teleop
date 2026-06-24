"""Round-trip tests for the CDR wire codecs.

The encoders and decoders are inverses; these lock down that contract (alignment,
endianness, length prefixes) without needing a live Adamo session.
"""

from __future__ import annotations

import numpy as np
import pytest

from adamo_isaac_teleop import cdr


def test_envelope_round_trip():
    payload = b"\x00\x01\x00\x00some-cdr-bytes"
    enc = cdr.encode_envelope("/controller/right", "geometry_msgs/msg/PoseStamped", payload)
    topic, type_name, body = cdr.decode_envelope(enc)
    assert topic == "/controller/right"
    assert type_name == "geometry_msgs/msg/PoseStamped"
    assert body == payload


def test_pose_stamped_round_trip():
    pos = [1.5, -2.25, 3.0]
    quat = [0.0, 0.0, 0.70710678, 0.70710678]  # xyzw
    out = cdr.decode_pose_stamped(cdr.encode_pose_stamped(pos, quat, frame_id="world"))
    assert out is not None
    got_pos, got_quat = out
    np.testing.assert_allclose(got_pos, pos)
    np.testing.assert_allclose(got_quat, quat)


def test_pose_array_round_trip():
    poses = np.array(
        [[0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0],
         [3.0, 4.0, 5.0, 0.5, 0.5, 0.5, 0.5]],
        np.float64,
    )
    out = cdr.decode_pose_array(cdr.encode_pose_array(poses))
    assert out is not None
    assert out.shape == (2, 7)
    np.testing.assert_allclose(out, poses)


def test_pose_array_empty():
    out = cdr.decode_pose_array(cdr.encode_pose_array(np.empty((0, 7))))
    assert out is not None
    assert out.shape == (0, 7)


def test_joy_round_trip():
    axes = [0.0, 0.5, -1.0, 0.25, 1.0]      # f32-exact values
    buttons = [0, 1, 0, 1]
    out = cdr.decode_joy(cdr.encode_joy(axes, buttons))
    assert out is not None
    got_axes, got_buttons = out
    np.testing.assert_allclose(got_axes, axes, rtol=0, atol=1e-7)
    assert got_buttons == buttons


def test_joint_state_round_trip():
    names = ["j1", "j2", "j3"]
    positions = [0.1, 0.2, 0.3]
    velocities = [1.0, -1.0, 0.0]
    out = cdr.decode_joint_state(
        cdr.encode_joint_state(names, positions, velocities))
    assert out is not None
    assert out["names"] == names
    np.testing.assert_allclose(out["positions"], positions)
    np.testing.assert_allclose(out["velocities"], velocities)
    assert out["efforts"] == []


@pytest.mark.parametrize("value", [True, False])
def test_bool_round_trip(value):
    assert cdr.decode_bool(cdr.encode_bool(value)) is value


def test_decoders_return_none_on_garbage():
    assert cdr.decode_pose_stamped(b"\x00\x01\x00\x00") is None
    assert cdr.decode_joy(b"\xff") is None
    assert cdr.decode_bool(b"") is None
