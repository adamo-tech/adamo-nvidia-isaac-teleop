"""Isaac Teleop device-representation seam over Adamo.

This is the wire between the **device interface** (capture side) and the **retargeting**
(robot side). It carries Isaac Teleop's own ``ControllerInput`` tensor group — the
canonical device representation (grip pose, aim pose, button/axis scalars, validity) —
so the retargeting side receives exactly what an Isaac device interface produces and
feeds it straight into the retargeter, with no intermediate format of our own.

``make_controller_input`` is the device-interface step: it turns raw operator motion
into Isaac's ``ControllerInput``. ``DeviceSink`` serializes that tensor group onto the
fabric; ``DeviceSource`` reconstructs the identical ``ControllerInput`` on the robot
side, ready to hand to :class:`~adamo_isaac_teleop.ControllerRetargetEngine`.

Compact little-endian binary (latency over self-description), published on
``{robot}/teleop/device/controller``; the handedness travels in the payload, so a
single topic carries any number of controllers.
"""

from __future__ import annotations

import struct
import threading
import time

import numpy as np
from isaacteleop.retargeting_engine.interface import TensorGroup
from isaacteleop.retargeting_engine.tensor_types import (
    ControllerInput,
    ControllerInputIndex,
)

# ControllerInput button/axis scalars, serialized in this fixed order. Each name maps to
# the ControllerInputIndex field it fills, so the seam carries Isaac's representation
# verbatim rather than a parallel copy of it.
_BUTTONS = (
    ("primary", ControllerInputIndex.PRIMARY_CLICK),
    ("secondary", ControllerInputIndex.SECONDARY_CLICK),
    ("thumbstick_click", ControllerInputIndex.THUMBSTICK_CLICK),
    ("thumbstick_x", ControllerInputIndex.THUMBSTICK_X),
    ("thumbstick_y", ControllerInputIndex.THUMBSTICK_Y),
    ("squeeze", ControllerInputIndex.SQUEEZE_VALUE),
    ("trigger", ControllerInputIndex.TRIGGER_VALUE),
)
BUTTONS = tuple(name for name, _ in _BUTTONS)


def _array(field) -> np.ndarray:
    """Read a tensor-group vector field as float64 (DLPack, like Isaac's retargeters)."""
    arr = np.from_dlpack(field) if hasattr(field, "__dlpack__") else np.asarray(field)
    return np.asarray(arr, np.float64).reshape(-1)


def _scalar(field) -> float:
    """Read a tensor-group scalar field as a Python float."""
    arr = np.from_dlpack(field) if hasattr(field, "__dlpack__") else np.asarray(field)
    return float(np.reshape(arr, -1)[0])


# --- device interface: build Isaac's ControllerInput from raw operator motion ---
def make_controller_input(
    grip_position,
    grip_orientation,
    aim_position=None,
    aim_orientation=None,
    buttons: dict[str, float] | None = None,
    grip_valid: bool = True,
    aim_valid: bool = True,
) -> TensorGroup:
    """Build Isaac Teleop's canonical ``ControllerInput`` from raw operator motion.

    This is the device-interface output: a ``ControllerInput`` tensor group the
    retargeter consumes directly (via :meth:`ControllerRetargetEngine.step_input`).
    Positions are ``(3,)`` and orientations ``(4,)`` xyzw; ``aim_*`` default to the grip
    pose when not supplied.
    """
    i = ControllerInputIndex
    gp = np.ascontiguousarray(grip_position, np.float32)
    gq = np.ascontiguousarray(grip_orientation, np.float32)
    ap = gp if aim_position is None else np.ascontiguousarray(aim_position, np.float32)
    aq = gq if aim_orientation is None else np.ascontiguousarray(aim_orientation, np.float32)
    btn = buttons or {}
    tg = TensorGroup(ControllerInput())
    tg[i.GRIP_POSITION] = gp
    tg[i.GRIP_ORIENTATION] = gq
    tg[i.GRIP_IS_VALID] = bool(grip_valid)
    tg[i.AIM_POSITION] = ap
    tg[i.AIM_ORIENTATION] = aq
    tg[i.AIM_IS_VALID] = bool(aim_valid)
    for name, idx in _BUTTONS:
        tg[idx] = float(btn.get(name, 0.0))
    return tg


def trigger_value(ci: TensorGroup) -> float:
    """Read the analog trigger (0..1) from a ``ControllerInput``."""
    return _scalar(ci[ControllerInputIndex.TRIGGER_VALUE])


# --- wire format ----------------------------------------------------------------
def _pack(side, grip_p, grip_q, aim_p, aim_q, button_vals, grip_valid, aim_valid) -> bytes:
    s = side.encode("utf-8")
    return (
        struct.pack("<B", len(s)) + s
        + struct.pack("<3f", *grip_p[:3])
        + struct.pack("<4f", *grip_q[:4])
        + struct.pack("<3f", *aim_p[:3])
        + struct.pack("<4f", *aim_q[:4])
        + struct.pack("<7f", *[float(v) for v in button_vals])
        + struct.pack("<2B", 1 if grip_valid else 0, 1 if aim_valid else 0)
    )


def _unpack(data: bytes):
    n = data[0]
    off = 1
    side = data[off:off + n].decode("utf-8"); off += n
    grip_p = struct.unpack_from("<3f", data, off); off += 12
    grip_q = struct.unpack_from("<4f", data, off); off += 16
    aim_p = struct.unpack_from("<3f", data, off); off += 12
    aim_q = struct.unpack_from("<4f", data, off); off += 16
    button_vals = struct.unpack_from("<7f", data, off); off += 28
    gv, av = struct.unpack_from("<2B", data, off)
    return side, grip_p, grip_q, aim_p, aim_q, button_vals, bool(gv), bool(av)


def encode_controller(side: str, ci: TensorGroup) -> bytes:
    """Serialize a ``ControllerInput`` (+ handedness) for the Adamo seam."""
    i = ControllerInputIndex
    return _pack(
        side,
        _array(ci[i.GRIP_POSITION]), _array(ci[i.GRIP_ORIENTATION]),
        _array(ci[i.AIM_POSITION]), _array(ci[i.AIM_ORIENTATION]),
        [_scalar(ci[idx]) for _, idx in _BUTTONS],
        _scalar(ci[i.GRIP_IS_VALID]) != 0.0,
        _scalar(ci[i.AIM_IS_VALID]) != 0.0,
    )


def decode_controller(data: bytes) -> tuple[str, TensorGroup]:
    """Reconstruct ``(side, ControllerInput)`` from the wire."""
    side, gp, gq, ap, aq, bvals, gv, av = _unpack(data)
    buttons = {name: bvals[k] for k, (name, _) in enumerate(_BUTTONS)}
    ci = make_controller_input(gp, gq, ap, aq, buttons, gv, av)
    return side, ci


class DeviceSink:
    """Device-interface output — publish the ``ControllerInput`` device representation."""

    def __init__(self, robot_name: str, session) -> None:
        self._key = f"{robot_name}/teleop/device/controller"
        self._pub = session.publisher(self._key, express=True, reliable=False)

    def controller(self, side: str, controller_input: TensorGroup) -> None:
        self._pub.put(encode_controller(side, controller_input))

    def close(self) -> None:
        pass


class DeviceSource:
    """Retargeting-side input — subscribe the ``ControllerInput`` device representation.

    Tracks whatever handedness(es) actually arrive; no side is hardcoded. ``controller``
    returns Isaac's ``ControllerInput`` tensor group, ready for the retargeter.
    """

    def __init__(self, robot_name: str, session) -> None:
        self._lock = threading.Lock()
        self._ctrl: dict[str, TensorGroup] = {}
        self._t: dict[str, float] = {}
        self._sub = session.subscribe(
            f"{robot_name}/teleop/device/controller", callback=self._cb)

    def _cb(self, sample) -> None:
        try:
            side, ci = decode_controller(bytes(sample.payload))
        except Exception:  # noqa: BLE001
            return
        with self._lock:
            self._ctrl[side] = ci
            self._t[side] = time.time()

    def controller(self, side: str) -> TensorGroup | None:
        with self._lock:
            return self._ctrl.get(side)

    def controller_t(self, side: str) -> float:
        with self._lock:
            return self._t.get(side, 0.0)

    def sides(self) -> list[str]:
        with self._lock:
            return list(self._ctrl)

    def close(self) -> None:
        self._sub = None
