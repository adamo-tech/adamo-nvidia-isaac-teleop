"""TeleopReceiver -- the Adamo web client device source (the device interface's input).

Decodes exactly what the **Adamo web client** publishes from a WebXR session and makes
it available to poll. This is the raw operator capture: a frontend running in the
headset's browser. (The robot side does NOT read this directly — the device interface
turns it into Isaac's ``ControllerInput`` and forwards that over the ``device`` seam;
see ``device.py``.)

Handedness is dynamic: whatever ``/controller/{h}`` or ``/hand/{h}`` the web client
sends is tracked under that key (``left`` / ``right`` / ``none`` / ...). Nothing about
the robot is assumed here.

    controller(side)  grip pose + buttons + trigger + fingertip (aim)
    head()            headset pose
    hand(side)        hand tracking -- (N, 7) joint poses [x,y,z, qx,qy,qz,qw]
    gamepad()         gamepad axes + buttons
    keyboard_held()   set of currently-held keys
    gello()           GELLO / leader-arm joint state
    clutch()          teleop clutch (bool)
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .cdr import (
    decode_bool,
    decode_envelope,
    decode_joint_state,
    decode_joy,
    decode_pose_array,
    decode_pose_stamped,
)


@dataclass
class Controller:
    position: np.ndarray | None = None       # (3,)  grip
    orientation: np.ndarray | None = None    # (4,) xyzw grip
    tip_position: np.ndarray | None = None   # (3,)  aim / fingertip
    tip_orientation: np.ndarray | None = None
    axes: list[float] = field(default_factory=list)
    buttons: list[int] = field(default_factory=list)
    trigger: float = 0.0
    t: float = 0.0


@dataclass
class Pose:
    position: np.ndarray | None = None
    orientation: np.ndarray | None = None
    t: float = 0.0


@dataclass
class Gamepad:
    axes: list[float] = field(default_factory=list)
    buttons: list[int] = field(default_factory=list)
    t: float = 0.0


def _joy_any(payload: bytes):
    """Decode a Joy sample whether enveloped or raw CDR (envelope-first)."""
    try:
        _, type_name, cdr = decode_envelope(payload)
        if "Joy" in type_name:
            r = decode_joy(cdr)
            if r is not None:
                return r
    except Exception:  # noqa: BLE001
        pass
    return decode_joy(payload)


def _jointstate_any(payload: bytes):
    try:
        _, type_name, cdr = decode_envelope(payload)
        if "JointState" in type_name:
            r = decode_joint_state(cdr)
            if r is not None:
                return r
    except Exception:  # noqa: BLE001
        pass
    return decode_joint_state(payload)


class TeleopReceiver:
    """Subscribe to the Adamo web client's control channels; expose latest decoded input."""

    TRIGGER_AXIS = 4  # button analog values follow the 4 thumbstick axes

    def __init__(self, robot_name: str, session) -> None:
        self._lock = threading.Lock()
        self._ctrl: dict = {}       # handedness -> Controller (populated as topics arrive)
        self._hands: dict = {}      # handedness -> (N, 7) ndarray
        self._head = Pose()
        self._gamepad = Gamepad()
        self._keys: dict = {}
        self._keys_t = 0.0
        self._gello = None
        self._gello_t = 0.0
        self._clutch = False
        r = robot_name
        self._subs = [
            session.subscribe(f"{r}/control/cdr/xr_tracking", callback=self._xr_cb),
            session.subscribe(f"{r}/control/cdr/joy", callback=self._joy_cb),
            session.subscribe(f"{r}/control/json/keyboard", callback=self._kb_cb),
            session.subscribe(f"{r}/control/cdr/joint_states", callback=self._gello_cb),
        ]
        print(f"[TeleopReceiver] web-client source on {r}/control/*", flush=True)

    # ---- subscription callbacks ----
    def _xr_cb(self, sample) -> None:
        try:
            topic, type_name, cdr = decode_envelope(bytes(sample.payload))
        except Exception:  # noqa: BLE001
            return
        now = time.time()
        parts = topic.strip("/").split("/")
        with self._lock:
            if type_name.endswith("PoseStamped"):
                d = decode_pose_stamped(cdr)
                if d is None:
                    return
                pos, quat = d
                if topic == "/head_pose":
                    self._head = Pose(pos, quat, now)
                elif len(parts) == 2 and parts[0] == "controller":           # /controller/{h}
                    c = self._ctrl.setdefault(parts[1], Controller())
                    c.position, c.orientation, c.t = pos, quat, now
                elif len(parts) == 3 and parts[0] == "controller" and parts[2] == "tip":
                    c = self._ctrl.setdefault(parts[1], Controller())
                    c.tip_position, c.tip_orientation = pos, quat
            elif type_name.endswith("PoseArray"):
                if len(parts) == 2 and parts[0] == "hand":                   # /hand/{h}
                    arr = decode_pose_array(cdr)
                    if arr is not None:
                        self._hands[parts[1]] = arr
            elif type_name.endswith("Joy"):
                if len(parts) == 3 and parts[0] == "controller" and parts[2] == "joy":
                    r = decode_joy(cdr)
                    if r is None:
                        return
                    axes, buttons = r
                    c = self._ctrl.setdefault(parts[1], Controller())
                    c.axes, c.buttons = axes, buttons
                    if len(axes) > self.TRIGGER_AXIS:
                        c.trigger = float(axes[self.TRIGGER_AXIS])
                    elif buttons:
                        c.trigger = float(buttons[0])
            elif type_name.endswith("Bool") and topic == "/teleop/clutch":
                b = decode_bool(cdr)
                if b is not None:
                    self._clutch = b

    def _joy_cb(self, sample) -> None:
        r = _joy_any(bytes(sample.payload))
        if r is None:
            return
        with self._lock:
            self._gamepad = Gamepad(r[0], r[1], time.time())

    def _kb_cb(self, sample) -> None:
        try:
            msg = json.loads(bytes(sample.payload).decode("utf-8"))
        except Exception:  # noqa: BLE001
            return
        key = str(msg.get("key", "")).lower()
        with self._lock:
            self._keys_t = time.time()
            if msg.get("action") == "down":
                self._keys[key] = time.time()
            elif msg.get("action") == "up":
                self._keys.pop(key, None)

    def _gello_cb(self, sample) -> None:
        js = _jointstate_any(bytes(sample.payload))
        if js is None:
            return
        with self._lock:
            self._gello = js
            self._gello_t = time.time()

    # ---- accessors (snapshots) ----
    def controller(self, side: str) -> Controller:
        with self._lock:
            c = self._ctrl.get(side)
            if c is None:
                return Controller()
            return Controller(
                position=None if c.position is None else c.position.copy(),
                orientation=None if c.orientation is None else c.orientation.copy(),
                tip_position=None if c.tip_position is None else c.tip_position.copy(),
                tip_orientation=None if c.tip_orientation is None else c.tip_orientation.copy(),
                axes=list(c.axes), buttons=list(c.buttons), trigger=c.trigger, t=c.t,
            )

    def controllers(self) -> list[str]:
        """Handedness keys seen so far (e.g. ['right'] or ['left','right'])."""
        with self._lock:
            return list(self._ctrl)

    def controller_t(self, side: str) -> float:
        with self._lock:
            c = self._ctrl.get(side)
            return 0.0 if c is None else c.t

    def head(self) -> Pose:
        with self._lock:
            h = self._head
            return Pose(None if h.position is None else h.position.copy(),
                        None if h.orientation is None else h.orientation.copy(), h.t)

    def hand(self, side: str) -> np.ndarray | None:
        with self._lock:
            a = self._hands.get(side)
            return None if a is None else a.copy()

    def gamepad(self) -> Gamepad:
        with self._lock:
            g = self._gamepad
            return Gamepad(list(g.axes), list(g.buttons), g.t)

    def keyboard_held(self, timeout: float = 0.3) -> set:
        now = time.time()
        with self._lock:
            return {k for k, t in self._keys.items() if now - t < timeout}

    def gello(self) -> dict | None:
        with self._lock:
            return None if self._gello is None else dict(self._gello)

    def clutch(self) -> bool:
        with self._lock:
            return self._clutch

    def close(self) -> None:
        self._subs.clear()
