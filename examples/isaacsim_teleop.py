#!/usr/bin/env python3
"""Isaac Sim teleop over Adamo, retargeted by NVIDIA Isaac Teleop.

Same mandated chain as the OpenArm example, with a *simulated* robot as the last hop:

    VR controller
      -> Adamo web client
      -> Isaac device interface             (decode web client -> Isaac ControllerInput)
      -> Adamo                              (device-representation seam)
      -> Isaac retargeting : ControllerRetargetEngine (Se3 retargeter -> ee_pose)
      -> Isaac Lab env.step()               (simulated arm)

Isaac Teleop's retargeter is the only path operator -> sim; there is no bypass.
Requires an x86 + RTX host with Isaac Lab installed (Isaac Sim needs an RTX GPU; it
does NOT run on Jetson). Run it there; the operator can be anywhere.

    ./isaaclab.sh -p examples/isaacsim_teleop.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0

TEMPLATE: not run in CI here (no Isaac Lab on the dev box). Tune two env-specific
things: the --task ACTION SPACE (7-vec IK-relative assumed) and the FRAME mapping
(`robot_from_retarget`).
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

# Isaac Lab requires launching the app before importing isaaclab.* -------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Adamo -> Isaac Teleop -> Isaac Lab")
parser.add_argument("--task", default="Isaac-Lift-Cube-Franka-IK-Rel-v0")
parser.add_argument("--side", default="right", choices=["right", "left"])
parser.add_argument("--sensitivity", type=float, default=3.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- now safe to import the rest ---------------------------------------------
import adamo
import gymnasium as gym
import isaaclab_tasks  # noqa: F401  (registers Isaac-* envs)
import torch
from isaaclab_tasks.utils import parse_env_cfg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from adamo_isaac_teleop import (
    ControllerRetargetEngine,
    DeviceSink,
    DeviceSource,
    TeleopReceiver,
    make_controller_input,
    trigger_value,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
API_KEY = os.environ.get("ADAMO_API_KEY")
WEB_ROBOT = "orin"
SEAM_ROBOT = "orin-isaac"
HZ = 60.0


def robot_from_retarget(delta: np.ndarray) -> np.ndarray:
    """Map an Isaac-retargeter ee-position delta into the sim env frame (TUNABLE)."""
    return np.array([-delta[2], -delta[0], delta[1]])


def device_interface(session_in, session_out, side, stop):
    """Device interface: decode the web client and publish Isaac's ControllerInput."""
    rx = TeleopReceiver(WEB_ROBOT, session_in)
    sink = DeviceSink(SEAM_ROBOT, session_out)
    dt = 1.0 / HZ
    while not stop.is_set():
        c = rx.controller(side)
        if c.position is not None:
            aim_p = c.tip_position if c.tip_position is not None else c.position
            aim_q = c.tip_orientation if c.tip_orientation is not None else c.orientation
            ci = make_controller_input(c.position, c.orientation, aim_p, aim_q,
                                       buttons={"trigger": c.trigger})
            sink.controller(side, ci)
        time.sleep(dt)


def main() -> int:
    if not API_KEY:
        raise SystemExit("set ADAMO_API_KEY (e.g. in a .env file -- see .env.example)")
    sess_op = adamo.connect(api_key=API_KEY, protocol="quic", mtls=True)
    sess_sim = adamo.connect(api_key=API_KEY, protocol="quic", mtls=True)
    stop = threading.Event()
    threading.Thread(target=device_interface, args=(sess_op, sess_op, args.side, stop),
                     daemon=True).start()

    rx = DeviceSource(SEAM_ROBOT, sess_sim)         # device seam -> retargeting input
    engine = ControllerRetargetEngine(args.side)    # Isaac retargeter

    env_cfg = parse_env_cfg(args.task, num_envs=1)
    env = gym.make(args.task, cfg=env_cfg)
    obs, _ = env.reset()
    device = env.unwrapped.device

    ee0 = None
    prev = np.zeros(3)
    print(f"TELEOP LIVE -> {args.task} via Isaac ControllerRetargetEngine", flush=True)
    try:
        while simulation_app.is_running():
            ci = rx.controller(args.side)
            ee = None if ci is None else engine.step_input(ci)
            if ee is not None:
                if ee0 is None:
                    ee0 = ee[:3].copy()
                rd = robot_from_retarget(ee[:3] - ee0)
            else:
                rd = prev
            step_delta = (rd - prev) * args.sensitivity
            prev = rd
            grip = 1.0 if (ci is not None and trigger_value(ci) > 0.5) else -1.0
            action = torch.tensor(
                [[step_delta[0], step_delta[1], step_delta[2], 0.0, 0.0, 0.0, grip]],
                dtype=torch.float32, device=device,
            )
            obs, _, terminated, truncated, _ = env.step(action)
            if bool(terminated) or bool(truncated):
                obs, _ = env.reset()
                ee0, prev = None, np.zeros(3)
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        stop.set()
        env.close()
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
