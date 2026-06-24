#!/usr/bin/env python3
"""OpenArm teleop over Adamo, retargeted by NVIDIA Isaac Teleop.

The full mandated chain, end to end, in one process:

    VR controller
      -> Adamo web client                          (operator, in a headset)
      -> Isaac device interface                    (decode web client -> Isaac ControllerInput)
      -> Adamo                                      (device-representation seam)
      -> Isaac retargeting : ControllerRetargetEngine (Se3 retargeter -> ee_pose)
      -> OpenArm                                    (damped-least-squares IK -> Waveshare CAN)

The robot side reuses the OpenArm project's proven ``ArmIKController`` (a stable
velocity IK with input smoothing and a home null-space pull) and homes to a bent-elbow
neutral pose before teleop begins.

Run on the robot host; drive it from the Adamo web client in a VR headset (controller
mode, robot name ``orin``):

    python examples/openarm_teleop.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from pathlib import Path

import adamo
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from adamo_isaac_teleop import (
    ControllerRetargetEngine,
    DeviceSink,
    DeviceSource,
    TeleopReceiver,
    make_controller_input,
    trigger_value,
)

OPENARM = Path(os.environ.get("OPENARM_DIR", Path.home() / "openarm_vr_teleop"))
sys.path.insert(0, str(OPENARM))          # for `import kinematics` / `can_motor`
sys.path.insert(0, str(OPENARM.parent))   # for `openarm_vr_teleop` as a package

from can_motor import OpenArmCAN
from kinematics import OpenArmKinematics
from openarm_vr_teleop.can_vr_teleop_mid import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    HOME_JOINTS_LEFT,
    HOME_JOINTS_RIGHT,
    ArmIKController,
    OneEuroFilter,
)

URDF = str(OPENARM / "urdf" / "openarm_bimanual_abs.urdf")
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
API_KEY = os.environ.get("ADAMO_API_KEY")

WEB_ROBOT = "orin"           # what the Adamo web client publishes to
SEAM_ROBOT = "orin-isaac"    # device-representation seam: device interface -> retargeting
CTRL_HZ = 60.0
HOME = {"right": HOME_JOINTS_RIGHT, "left": HOME_JOINTS_LEFT}
DEFAULT_CAN = {"right": "waveshare:0", "left": "waveshare:1"}


# --------------------------------------------------------------------------
# Device interface: web client -> Isaac ControllerInput on the device seam
# --------------------------------------------------------------------------
def device_interface(session_in, session_out, sides, stop: threading.Event) -> None:
    rx = TeleopReceiver(WEB_ROBOT, session_in)
    sink = DeviceSink(SEAM_ROBOT, session_out)
    print(f"[device] {WEB_ROBOT} web client -> {SEAM_ROBOT} device seam", flush=True)
    dt = 1.0 / CTRL_HZ
    while not stop.is_set():
        for s in sides:
            c = rx.controller(s)
            if c.position is None:
                continue
            aim_p = c.tip_position if c.tip_position is not None else c.position
            aim_q = c.tip_orientation if c.tip_orientation is not None else c.orientation
            ci = make_controller_input(c.position, c.orientation, aim_p, aim_q,
                                       buttons={"trigger": c.trigger})
            sink.controller(s, ci)
        time.sleep(dt)


# --------------------------------------------------------------------------
# OpenArm hardware helpers
# --------------------------------------------------------------------------
def robust_read(arm, tries: int = 30):
    pos = np.zeros(7)
    seen = [False] * 7
    for _ in range(tries):
        arm.refresh()
        for i in range(7):
            if arm.states[i].valid:
                pos[i] = arm.states[i].position
                seen[i] = True
        if all(seen):
            return pos, seen
        time.sleep(0.04)
    return pos, seen


def cleanup(arm) -> None:
    for fn in (arm.disable_all, arm.close):
        try:
            fn()
        except Exception:  # noqa: BLE001
            pass


def home_arm(arm, start: np.ndarray, home: np.ndarray, kp: float, kd: float,
             duration: float = 2.0) -> np.ndarray:
    """Cosine ease-in-out ramp from the current pose to the bent-elbow home."""
    steps = int(duration * CTRL_HZ)
    for i in range(steps):
        a = 0.5 * (1.0 - math.cos(math.pi * (i + 1) / steps))
        j = start + a * (home - start)
        arm.set_joint_positions(j, kp=kp * 0.5, kd=kd, process_responses=False)
        arm._process_responses()
        time.sleep(1.0 / CTRL_HZ)
    return home.copy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--right-can", default=DEFAULT_CAN["right"])
    ap.add_argument("--left-can", default=DEFAULT_CAN["left"])
    ap.add_argument("--kp", type=float, default=35.0)
    ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--max-vel", type=float, default=6.0, help="joint velocity cap (rad/s)")
    ap.add_argument("--smooth", type=float, default=6.0,
                    help="1 Euro min_cutoff; higher = snappier/less lag, lower = smoother")
    args = ap.parse_args()
    if not API_KEY:
        print("set ADAMO_API_KEY (e.g. in a .env file -- see .env.example)", flush=True)
        return 1
    sides = ["right", "left"]
    cans = {"right": args.right_can, "left": args.left_can}

    print("connecting to Adamo (operator + robot participants)...", flush=True)
    sess_op = adamo.connect(api_key=API_KEY, protocol="quic", mtls=True)
    sess_robot = adamo.connect(api_key=API_KEY, protocol="quic", mtls=True)

    stop = threading.Event()
    threading.Thread(target=device_interface, args=(sess_op, sess_op, sides, stop),
                     daemon=True).start()
    rx = DeviceSource(SEAM_ROBOT, sess_robot)

    # ---- enable + home each arm that comes up ----
    arms, kins, cur = {}, {}, {}
    for s in sides:
        kin = OpenArmKinematics(URDF, f"openarm_{s}_")
        print(f"[{s}] ENABLE on {cans[s]}", flush=True)
        arm = OpenArmCAN(cans[s])
        arm.enable_all()
        time.sleep(0.1)
        start, seen = robust_read(arm)
        print(f"[{s}] joints {sum(seen)}/7  pos={np.round(start, 3).tolist()}")
        if not all(seen):
            missing = [i + 1 for i, v in enumerate(seen) if not v]
            print(f"[{s}] skipping: joints {missing} never reported on {cans[s]}.")
            cleanup(arm)
            continue
        print(f"[{s}] homing to bent-elbow neutral...", flush=True)
        cur[s] = home_arm(arm, start, HOME[s], args.kp, args.kd)
        arm.set_gripper(GRIPPER_OPEN, kp=args.kp, kd=args.kd)
        arms[s], kins[s] = arm, kin
    if not arms:
        print("no arms came up; aborting.")
        stop.set()
        return 1
    sides = list(arms)

    # ---- wait for the operator, then calibrate per arm ----
    print("waiting for the operator -- move a controller in the web client (VR)...", flush=True)
    t0 = time.time()
    while all(rx.controller(s) is None for s in sides):
        if time.time() - t0 > 300:
            print("no operator input; aborting.")
            stop.set()
            return 1
        time.sleep(0.2)

    engines, iks, euros = {}, {}, {}
    for s in sides:
        engine = ControllerRetargetEngine(s)
        # The retargeter can return None for a few frames before it locks on, even
        # with a controller present -- warm it up before calibrating off its pose.
        ee, t0 = None, time.time()
        while ee is None and time.time() - t0 < 5.0:
            ci = rx.controller(s)
            if ci is not None:
                ee = engine.step_input(ci)
            if ee is None:
                time.sleep(1.0 / CTRL_HZ)
        if ee is None:
            print(f"[{s}] no retargeted pose; skipping this arm.", flush=True)
            continue
        engines[s] = engine
        iks[s] = ArmIKController(kins[s], side=s, position_scale=1.0,
                                 max_joint_velocity=args.max_vel)
        iks[s].calibrate(cur[s], ee[:3])              # position-only (no orientation)
        euros[s] = OneEuroFilter(np.asarray(ee[:3]), min_cutoff=args.smooth)
    if not iks:
        print("no arm could be calibrated; aborting.")
        stop.set()
        for arm in arms.values():
            cleanup(arm)
        return 1

    dt = 1.0 / CTRL_HZ
    print(f"TELEOP LIVE on {', '.join(iks)} -- via Isaac ControllerRetargetEngine. Ctrl+C to stop.",
          flush=True)
    last_dbg = 0.0
    prev: dict = {}
    try:
        while True:
            now = time.time()
            for s in iks:
                ci = rx.controller(s)
                if ci is None:
                    continue
                # gripper: trigger -> open/close
                trig = float(np.clip(trigger_value(ci), 0.0, 1.0))
                grip_q = GRIPPER_OPEN + trig * (GRIPPER_CLOSED - GRIPPER_OPEN)
                arms[s].set_gripper(grip_q, kp=args.kp, kd=args.kd)
                # arm: retarget the controller pose -> DLS velocity IK
                ee = engines[s].step_input(ci)
                ee_pos = None if ee is None else euros[s](np.asarray(ee[:3]), now)
                cmd = None if ee_pos is None else iks[s].compute(ee_pos, dt)
                step = 0.0
                if cmd is not None:
                    step = float(np.max(np.abs(cmd - prev.get(s, cmd))))  # true per-frame step
                    prev[s] = cmd.copy()
                    arms[s].set_joint_positions(cmd, kp=args.kp, kd=args.kd,
                                                process_responses=False)
                    cur[s] = cmd
                arms[s]._process_responses()
                if now - last_dbg > 0.5 and cmd is not None:
                    print(f"[{s}] ee={np.round(ee_pos, 3).tolist()}  step={step:.3f}  "
                          f"trig={trig:.2f} grip={grip_q:.2f}", flush=True)
            if now - last_dbg > 0.5:
                last_dbg = now
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        stop.set()
        print("disabling motors", flush=True)
        for arm in arms.values():
            cleanup(arm)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
