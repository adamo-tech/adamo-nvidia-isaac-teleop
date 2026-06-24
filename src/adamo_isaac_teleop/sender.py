"""TeleopSender -- publish operator teleop over Adamo (the mirror of TeleopReceiver).

Lets *any* frontend feed the same Adamo pipeline that a robot-side ``TeleopReceiver``
consumes: a third-party Isaac Teleop / IsaacSim app, a custom operator UI, or a test
harness. It publishes on the same topics and in the same wire format the Adamo web
client uses, so the two ends interoperate.

    sender.controller(side, position, orientation_xyzw, trigger=...)
    sender.head(position, orientation_xyzw)
    sender.hand(side, joints)            # (N, 7) joint poses
    sender.gamepad(axes, buttons)
    sender.keyboard(key, action)         # "down" / "up"
    sender.joints(names, positions)      # GELLO / leader-arm joint state
    sender.clutch(engaged)
"""

from __future__ import annotations

import json

from .cdr import (
    encode_bool,
    encode_envelope,
    encode_joint_state,
    encode_joy,
    encode_pose_array,
    encode_pose_stamped,
)


class TeleopSender:
    """Publish teleop input for ``robot_name`` over an Adamo ``session``."""

    def __init__(self, robot_name: str, session) -> None:
        opts = dict(express=True, reliable=False)  # real-time, drop-on-congestion
        self._xr = session.publisher(f"{robot_name}/control/cdr/xr_tracking", **opts)
        self._joy = session.publisher(f"{robot_name}/control/cdr/joy", **opts)
        self._kb = session.publisher(f"{robot_name}/control/json/keyboard", **opts)
        self._js = session.publisher(f"{robot_name}/control/cdr/joint_states", **opts)

    def controller(self, side: str, position, orientation_xyzw,
                   trigger: float | None = None, buttons: list[int] | None = None) -> None:
        self._xr.put(encode_envelope(
            f"/controller/{side}", "geometry_msgs/msg/PoseStamped",
            encode_pose_stamped(position, orientation_xyzw)))
        if trigger is not None or buttons:
            axes = [0.0, 0.0, 0.0, 0.0, float(trigger or 0.0)]  # trigger at axes[4]
            self._xr.put(encode_envelope(
                f"/controller/{side}/joy", "sensor_msgs/msg/Joy",
                encode_joy(axes, buttons or [])))

    def head(self, position, orientation_xyzw) -> None:
        self._xr.put(encode_envelope(
            "/head_pose", "geometry_msgs/msg/PoseStamped",
            encode_pose_stamped(position, orientation_xyzw)))

    def hand(self, side: str, joints) -> None:
        """joints: (N, 7) [x,y,z, qx,qy,qz,qw] per joint."""
        self._xr.put(encode_envelope(
            f"/hand/{side}", "geometry_msgs/msg/PoseArray", encode_pose_array(joints)))

    def gamepad(self, axes, buttons=()) -> None:
        self._joy.put(encode_envelope(
            "/joy", "sensor_msgs/msg/Joy", encode_joy(list(axes), list(buttons))))

    def keyboard(self, key: str, action: str = "down", code: str = "") -> None:
        self._kb.put(json.dumps({"key": key, "code": code or key, "action": action}).encode())

    def joints(self, names, positions, velocities=(), efforts=()) -> None:
        self._js.put(encode_envelope(
            "/joint_states", "sensor_msgs/msg/JointState",
            encode_joint_state(names, positions, velocities, efforts)))

    def clutch(self, engaged: bool) -> None:
        self._xr.put(encode_envelope("/teleop/clutch", "std_msgs/msg/Bool", encode_bool(engaged)))

    def close(self) -> None:
        pass
