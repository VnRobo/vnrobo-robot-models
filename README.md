# VnRobo Robot Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.x-blue)](https://mujoco.org)

MJCF simulation models for VnRobo's open-source robot lineup.

## Robots

### VNR-WH1 — Wheeled Humanoid

> "Practical for factories, agile in the lab"

An 18-DOF wheeled humanoid designed for indoor manipulation tasks.

| Spec | Value |
|------|-------|
| Height | ~1.35 m |
| Total mass | ~57 kg |
| Controllable DOF | 18 |
| Base | 3-wheel omnidirectional |
| Arms | 6-DOF × 2 + 2-finger gripper |
| Head | 2-DOF (pan / tilt) |
| Sensors (sim) | IMU, RGB-D camera, joint encoders |

```
         ┌──────────────────┐
         │  Head (pan/tilt) │   cam_rgb ● cam_depth ●
         └────────┬─────────┘
    ┌─────────────┴─────────────┐
    │      Chest / Torso        │
    │ [6-DOF arm]   [6-DOF arm] │   ← amber highlights
    │ [gripper]       [gripper] │
    └─────────────┬─────────────┘
         ┌────────┴────────┐
         │  Mobile Base    │   ← dark grey platform
         │   ⊙  ⊙  ⊙     │   3 omnidirectional wheels
         └─────────────────┘
```

**Kinematic chain (18 actuated joints):**

| Group | Joints |
|-------|--------|
| Wheels | `wheel_f`, `wheel_bl`, `wheel_br` (velocity ctrl) |
| Head | `head_pan`, `head_tilt` |
| Left arm | `l_sp`, `l_sr`, `l_sy`, `l_ep`, `l_wp`, `l_wr` |
| Left gripper | `l_grip` (coupled 2-finger) |
| Right arm | `r_sp`, `r_sr`, `r_sy`, `r_ep`, `r_wp`, `r_wr` |
| Right gripper | `r_grip` (coupled 2-finger) |

## Quick Start

```bash
pip install "mujoco>=3.0"

# Validate model
python examples/test_joints.py

# Interactive viewer (requires display)
python examples/view_model.py
python examples/view_model.py --scene   # with manipulation objects
```

## Repository Structure

```
vnrobo-robot-models/
├── vnr_wh1/
│   ├── vnr_wh1.xml          ← main MJCF (primitive geometry v0.1)
│   ├── scenes/
│   │   └── default.xml      ← flat floor + pick-place objects
│   └── assets/meshes/       ← STL meshes (v0.2, coming soon)
├── examples/
│   ├── test_joints.py       ← smoke tests
│   └── view_model.py        ← MuJoCo viewer launcher
└── vnr_h1/                  ← bipedal humanoid (coming in v0.3)
```

## Roadmap

| Version | Milestone |
|---------|-----------|
| **v0.1** (current) | Primitive geometry, full kinematic tree, CI |
| v0.2 | STL visual meshes (Fusion 360 export), warehouse scene |
| v0.3 | Physics-tuned inertia, force/torque sensors, URDF export |
| v0.4 | RL locomotion + manipulation baselines in `rl-robotics-starter` |
| v0.5 | VNR-H1 bipedal model |
| v1.0 | MuJoCo Menagerie submission |

## Related Repos

- [`vnrobo-agent`](https://github.com/VnRobo/vnrobo-agent) — Python SDK to stream robot telemetry
- [`rl-robotics-starter`](https://github.com/VnRobo/rl-robotics-starter) — RL training baselines
- [`ros2-fleet-bridge`](https://github.com/VnRobo/ros2-fleet-bridge) — ROS 2 fleet monitoring

## Monitor your robots for free

Deploy on real hardware? Monitor with [VnRobo Fleet Monitor](https://app.vnrobo.com) — free for up to 3 robots.

## License

MIT — see [LICENSE](LICENSE).
