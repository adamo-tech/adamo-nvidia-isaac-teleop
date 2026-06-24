"""Adamo × NVIDIA Isaac Teleop.

Teleoperate a robot over the Adamo real-time fabric, with **NVIDIA Isaac Teleop doing
the retargeting**. The operator's input (captured by the Adamo web client) is carried
over Adamo; on the robot side it is retargeted by Isaac Teleop's engines into an
end-effector pose. There is deliberately no direct-mapping shortcut — Isaac is the
only path from operator to robot.

    TeleopReceiver / TeleopSender   -- decode / publish control over Adamo (transport)
    ControllerRetargetEngine        -- Isaac Se3 retargeter: controller pose -> ee_pose
    HandRetargetEngine              -- Isaac retargeter: 26-joint hand -> ee_pose + gripper

Inverse kinematics and actuation are left to the robot side — see ``examples/`` for a
complete OpenArm integration and an Isaac Sim one.
"""

from __future__ import annotations

from . import cdr
from .device import DeviceSink, DeviceSource, make_controller_input, trigger_value
from .receiver import Controller, Gamepad, Pose, TeleopReceiver
from .retarget import ControllerRetargetEngine, HandRetargetEngine
from .sender import TeleopSender

__all__ = [
    # web-client device source (raw operator capture)
    "TeleopReceiver",
    "TeleopSender",
    "Controller",
    "Pose",
    "Gamepad",
    # Isaac device-representation seam (device interface -> Adamo -> retargeting)
    "DeviceSink",
    "DeviceSource",
    "make_controller_input",
    "trigger_value",
    # Isaac retargeting (robot side)
    "ControllerRetargetEngine",
    "HandRetargetEngine",
    "cdr",
]

__version__ = "0.1.0"
