"""Isaac Teleop retargeting graph, driven headless (no TeleopSession / OpenXR).

On a Jetson there is no OpenXR runtime, and ``TeleopSession.__enter__`` hard-requires
one (it calls ``oxr.OpenXRSession(...)``, which fails with ``xrCreateInstance failed``).
But the retargeters themselves are pure numpy/scipy — so we build the graph and
execute it directly via ``execute_pipeline()``, skipping the device/session layer
entirely. That is what lets the retarget side run on the Orin with no CloudXR.

The graph::

    ValueInput("hand_input", HandInput)
        |-- Se3AbsRetargeter   -> "ee_pose"          (7,) position + quat(xyzw)
        '-- GripperRetargeter  -> "gripper_command"  scalar, -1 closed .. +1 open
                                   '-- OutputCombiner

Feed it a 26-joint hand each frame; get back an end-effector pose + gripper.
The hand data is exactly the Isaac Teleop ``HandInput`` layout — the same thing
Adamo carries over the wire from the WebXR capture running in the headset browser.
"""

from __future__ import annotations

import numpy as np
from isaacteleop.retargeters.gripper_retargeter import (
    GripperRetargeter,
    GripperRetargeterConfig,
)
from isaacteleop.retargeters.se3_retargeter import (
    Se3AbsRetargeter,
    Se3RetargeterConfig,
)
from isaacteleop.retargeting_engine.interface import (
    OutputCombiner,
    TensorGroup,
    ValueInput,
)
from isaacteleop.retargeting_engine.tensor_types import (
    NUM_HAND_JOINTS,
    ControllerInput,
    ControllerInputIndex,
    HandInput,
    HandInputIndex,
)

NUM_JOINTS = int(NUM_HAND_JOINTS)  # 26, OpenXR XrHandJointEXT order

EeAndGripper = tuple[np.ndarray, float]


class HandRetargetEngine:
    """Headless Isaac Teleop hand -> (ee_pose, gripper) retargeter.

    Build once, then call :meth:`step` per frame with a 26-joint hand.
    """

    def __init__(self, side: str = "right") -> None:
        device = f"hand_{side}"
        self._hand_in = ValueInput("hand_input", HandInput())
        se3 = Se3AbsRetargeter(Se3RetargeterConfig(input_device=device), name="ee")
        grip = GripperRetargeter(GripperRetargeterConfig(hand_side=side), name="grip")
        se3_sub = se3.connect({device: self._hand_in.output(ValueInput.VALUE)})
        grip_sub = grip.connect({device: self._hand_in.output(ValueInput.VALUE)})
        self._pipe = OutputCombiner(
            {
                "ee_pose": se3_sub.output("ee_pose"),
                "gripper": grip_sub.output("gripper_command"),
            }
        )
        self._leaf = self._hand_in.name      # "hand_input"
        self._input_key = ValueInput.VALUE   # "value"

    def step(
        self,
        joint_positions: np.ndarray,     # (26, 3) float32, meters
        joint_orientations: np.ndarray,  # (26, 4) float32, quaternion xyzw
        joint_radii: np.ndarray,         # (26,)   float32, meters
        joint_valid: np.ndarray,         # (26,)   uint8, 1 = tracked
    ) -> EeAndGripper | None:
        """Retarget one hand frame.

        Returns ``(ee_pose, gripper)`` where ``ee_pose`` is a ``(7,)`` array of
        position (xyz) + quaternion (xyzw), and ``gripper`` is a float in
        ``[-1, +1]`` (-1 closed, +1 open). Returns ``None`` if the hand is absent.
        """
        tg = TensorGroup(HandInput())
        tg[HandInputIndex.JOINT_POSITIONS] = np.ascontiguousarray(joint_positions, dtype=np.float32)
        tg[HandInputIndex.JOINT_ORIENTATIONS] = np.ascontiguousarray(
            joint_orientations, dtype=np.float32)
        tg[HandInputIndex.JOINT_RADII] = np.ascontiguousarray(joint_radii, dtype=np.float32)
        tg[HandInputIndex.JOINT_VALID] = np.ascontiguousarray(joint_valid, dtype=np.uint8)

        out = self._pipe.execute_pipeline({self._leaf: {self._input_key: tg}})
        ee, grip = out["ee_pose"], out["gripper"]
        if getattr(ee, "is_none", False):
            return None
        ee_pose = np.asarray(ee[0], dtype=np.float32).copy()
        gripper = float(np.asarray(grip[0]))
        return ee_pose, gripper


class ControllerRetargetEngine:
    """Headless Isaac Teleop VR-controller -> ee_pose retargeter.

    Drives the Se3AbsRetargeter from a controller grip pose (the same data the
    Adamo web client publishes). ``ValueInput`` passes the whole ``ControllerInput``
    through, so we fill every field each frame (grip + aim + scalar buttons).
    """

    def __init__(self, side: str = "right") -> None:
        device = f"controller_{side}"
        self._in = ValueInput("controller_input", ControllerInput())
        se3 = Se3AbsRetargeter(Se3RetargeterConfig(input_device=device), name="ee")
        sub = se3.connect({device: self._in.output(ValueInput.VALUE)})
        self._pipe = OutputCombiner({"ee_pose": sub.output("ee_pose")})
        self._leaf = self._in.name
        self._key = ValueInput.VALUE

    def step(self, grip_pos, grip_quat_xyzw, valid: bool = True) -> np.ndarray | None:
        """Controller grip pose -> ee_pose (7,) pos+quat, or None if absent."""
        i = ControllerInputIndex
        gp = np.ascontiguousarray(grip_pos, np.float32)
        gq = np.ascontiguousarray(grip_quat_xyzw, np.float32)
        tg = TensorGroup(ControllerInput())
        tg[i.GRIP_POSITION] = gp
        tg[i.GRIP_ORIENTATION] = gq
        tg[i.GRIP_IS_VALID] = bool(valid)
        tg[i.AIM_POSITION] = gp
        tg[i.AIM_ORIENTATION] = gq
        tg[i.AIM_IS_VALID] = bool(valid)
        for f in (i.PRIMARY_CLICK, i.SECONDARY_CLICK, i.THUMBSTICK_CLICK,
                  i.THUMBSTICK_X, i.THUMBSTICK_Y, i.SQUEEZE_VALUE, i.TRIGGER_VALUE):
            tg[f] = 0.0
        out = self._pipe.execute_pipeline({self._leaf: {self._key: tg}})
        ee = out["ee_pose"]
        if getattr(ee, "is_none", False):
            return None
        return np.asarray(ee[0], dtype=np.float32).copy()
