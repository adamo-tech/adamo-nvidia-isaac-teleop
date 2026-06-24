# OpenArm example

`openarm_teleop.py` teleoperates a bimanual **OpenArm** (Damiao motors over a
**Waveshare USB-CAN-FD** adapter, Placo IK) from a VR headset over Adamo.

## Setup

```bash
pip install -e ".[openarm]"     # from the repo root — adds Placo + python-can
```

- The OpenArm driver must be importable: `kinematics.py` (Placo IK), `can_motor.py`,
  and `urdf/`. Point at it with `OPENARM_DIR` (default `~/openarm_vr_teleop`).
- Power on the OpenArm and connect its Waveshare adapter (channels `waveshare:0` and
  `waveshare:1`).

## Run

```bash
python examples/openarm_teleop.py
```

Connect to the robot in the Adamo web client (robot name `orin`), enter the immersive
VR view, and move a controller. Each arm holds its current pose until input arrives,
then follows. **Ctrl+C** disables the motors.

| flag | default | |
|---|---|---|
| `--right-can` / `--left-can` | `waveshare:0` / `waveshare:1` | CAN channel per arm |
| `--kp` / `--kd` | `35` / `1` | motor gains |

Environment: `ADAMO_API_KEY`, `OPENARM_DIR`.

## Tuning the retargeted frame

Isaac's retargeter outputs an end-effector pose in its own frame; `robot_from_retarget()`
at the top of `openarm_teleop.py` maps that into the OpenArm base frame. If an axis moves
the wrong way, flip its sign there.
