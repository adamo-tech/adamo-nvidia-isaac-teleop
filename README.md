# Adamo × NVIDIA Isaac Teleop

Teleoperate a robot from anywhere over the internet. [Adamo](https://adamohq.com)
carries the operator's motion over a low-latency Zenoh fabric;
[NVIDIA Isaac Teleop](https://github.com/NVIDIA/IsaacTeleop) retargets it onto the
robot. An operator in a VR headset moves naturally and the robot follows — across a WAN.

```
 Teleoperator (VR)
   → Isaac Teleop · device interface
   → Adamo
   → Isaac Teleop · retargeting
   → Robot
```

## How it works

Isaac Teleop separates a **device interface** (capture human motion) from a
**retargeting interface** (map it to robot commands). Adamo carries data across that
seam, so the two halves can run on different machines, anywhere:

- **Isaac Teleop's device interface** captures the teleoperator's motion and publishes
  it over Adamo.
- **Isaac Teleop's retargeters** receive it over Adamo and turn it into an end-effector
  pose. They run headless via `execute_pipeline()` — no GPU or OpenXR runtime needed —
  so this half runs on an edge device such as a Jetson.
- You take the retargeted pose and drive the robot with your own IK and actuation.

## Install

Python 3.10.

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Examples

Both run the full pipeline; drive either from the Adamo web client in a VR headset
(controller mode).

### OpenArm — real robot

A bimanual OpenArm over a Waveshare USB-CAN-FD adapter, with Placo IK. See
[`examples/README.md`](./examples/README.md) for hardware setup.

```bash
.venv/bin/pip install -e ".[openarm]"
python examples/openarm_teleop.py
```

### Isaac Sim — simulated robot

Drives a simulated arm in an Isaac Lab environment. Requires an x86 + RTX host with
Isaac Lab installed.

```bash
./isaaclab.sh -p examples/isaacsim_teleop.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0
```

## Use in your own robot

The library gives you the operator's input over Adamo and Isaac's retargeting; you
supply IK and actuation:

```python
import adamo
from adamo_isaac_teleop import TeleopReceiver, ControllerRetargetEngine

session = adamo.connect(api_key=..., protocol="quic", mtls=True)
rx = TeleopReceiver("my-robot", session)
engine = ControllerRetargetEngine("right")

while True:                                       # your control loop
    c = rx.controller("right")
    if c.position is None:
        continue
    ee = engine.step(c.position, c.orientation)   # -> ee_pose (7,), or None
    if ee is not None:
        joints = your_ik(ee)                      # your robot's IK
        your_robot.command(joints)                # your actuation
```

`HandRetargetEngine` does the same for 26-joint hand tracking (→ ee_pose + gripper).

### Publishing from a custom frontend

Not using the Adamo web client? Publish the operator's input yourself with
`TeleopSender` — the robot side decodes it identically:

```python
import adamo
from adamo_isaac_teleop import TeleopSender

session = adamo.connect(api_key=..., protocol="quic", mtls=True)
tx = TeleopSender("my-robot", session)

while True:                                   # your capture loop
    pos, quat = read_controller()             # position (3,), quaternion xyzw (4,)
    tx.controller("right", pos, quat, trigger=grip)   # also: head(), hand(), gamepad(), keyboard()
```

## Repository layout

```
src/adamo_isaac_teleop/
  cdr.py        the control wire format (decode + encode)
  receiver.py   TeleopReceiver — decode the operator's input over Adamo
  sender.py     TeleopSender — publish the operator's input over Adamo
  device.py     the device-interface ↔ retargeting seam over Adamo
  retarget.py   Isaac Teleop retargeters (headless)
examples/
  openarm_teleop.py    real OpenArm (Placo IK + Waveshare CAN)
  isaacsim_teleop.py   Isaac Lab environment (x86 + RTX)
```

## Isaac Teleop data format (reference)

Typed tensor groups, CPU float32, ~60 Hz:
- inputs: `ControllerInput` (grip pose + buttons) or `HandInput`
  (26 joints × {pos(3), quat-xyzw(4), radius, valid});
- outputs: `ee_pose` `(7,)` position + quaternion, `gripper` scalar.

## License

MIT — see [LICENSE](./LICENSE). Isaac Teleop, a dependency, is Apache-2.0.
