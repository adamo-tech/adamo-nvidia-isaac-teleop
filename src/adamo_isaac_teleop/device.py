"""Isaac Teleop device-representation seam over Adamo.

This is the wire between **Isaac Teleop #1 (device interface)** and **Isaac Teleop #2
(retargeting)**. It carries Isaac Teleop's canonical *device* representation — the
fields of ``ControllerInput`` (grip pose, aim pose, buttons) — rather than any
particular frontend's format. So the retargeting side accepts anything an Isaac device
interface produces, regardless of how the operator was originally captured.

Compact little-endian binary (latency over self-description), published on
``{robot}/teleop/device/controller``; the handedness travels in the payload, so a
single topic carries any number of controllers.
"""

from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Isaac ControllerInput button/axis scalars, in a fixed order.
BUTTONS = ("primary", "secondary", "thumbstick_click",
           "thumbstick_x", "thumbstick_y", "squeeze", "trigger")


@dataclass
class ControllerDevice:
    """Isaac Teleop ``ControllerInput`` fields for one controller."""

    side: str
    grip_position: np.ndarray        # (3,)
    grip_orientation: np.ndarray     # (4,) xyzw
    aim_position: np.ndarray         # (3,)
    aim_orientation: np.ndarray      # (4,) xyzw
    buttons: Dict[str, float] = field(default_factory=dict)
    grip_valid: bool = True
    aim_valid: bool = True
    t: float = 0.0


def encode_controller(dev: ControllerDevice) -> bytes:
    side = dev.side.encode("utf-8")
    return (
        struct.pack("<B", len(side)) + side
        + struct.pack("<3f", *dev.grip_position)
        + struct.pack("<4f", *dev.grip_orientation)
        + struct.pack("<3f", *dev.aim_position)
        + struct.pack("<4f", *dev.aim_orientation)
        + struct.pack("<7f", *[float(dev.buttons.get(b, 0.0)) for b in BUTTONS])
        + struct.pack("<2B", 1 if dev.grip_valid else 0, 1 if dev.aim_valid else 0)
    )


def decode_controller(data: bytes) -> ControllerDevice:
    n = data[0]
    off = 1
    side = data[off:off + n].decode("utf-8"); off += n
    gp = np.array(struct.unpack_from("<3f", data, off), np.float64); off += 12
    gq = np.array(struct.unpack_from("<4f", data, off), np.float64); off += 16
    ap = np.array(struct.unpack_from("<3f", data, off), np.float64); off += 12
    aq = np.array(struct.unpack_from("<4f", data, off), np.float64); off += 16
    bvals = struct.unpack_from("<7f", data, off); off += 28
    gv, av = struct.unpack_from("<2B", data, off)
    return ControllerDevice(side, gp, gq, ap, aq, dict(zip(BUTTONS, bvals)), bool(gv), bool(av))


class DeviceSink:
    """Isaac Teleop #1 output — publish the device representation over Adamo."""

    def __init__(self, robot_name: str, session) -> None:
        self._key = f"{robot_name}/teleop/device/controller"
        self._pub = session.publisher(self._key, express=True, reliable=False)

    def controller(self, dev: ControllerDevice) -> None:
        self._pub.put(encode_controller(dev))

    def close(self) -> None:
        pass


class DeviceSource:
    """Isaac Teleop #2 input — subscribe the device representation over Adamo.

    Tracks whatever handedness(es) actually arrive; no side is hardcoded.
    """

    def __init__(self, robot_name: str, session) -> None:
        self._lock = threading.Lock()
        self._ctrl: Dict[str, ControllerDevice] = {}
        self._sub = session.subscribe(
            f"{robot_name}/teleop/device/controller", callback=self._cb)

    def _cb(self, sample) -> None:
        try:
            dev = decode_controller(bytes(sample.payload))
            dev.t = time.time()
        except Exception:  # noqa: BLE001
            return
        with self._lock:
            self._ctrl[dev.side] = dev

    def controller(self, side: str) -> Optional[ControllerDevice]:
        with self._lock:
            return self._ctrl.get(side)

    def sides(self) -> List[str]:
        with self._lock:
            return list(self._ctrl)

    def close(self) -> None:
        self._sub = None
