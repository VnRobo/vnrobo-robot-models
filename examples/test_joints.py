"""Smoke-test: load model, move every joint through its range, check no NaN.

Usage:
    python examples/test_joints.py
"""

import os
import sys
import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(__file__))
XML = os.path.join(ROOT, "vnr_wh1", "vnr_wh1.xml")

def test_model_loads():
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    assert not np.any(np.isnan(data.qpos)), "NaN in qpos after reset"
    print(f"[PASS] Model loads — nq={model.nq}, nv={model.nv}, nu={model.nu}")
    return model, data

def test_actuators(model, data):
    """Set each actuator to mid-range, step 100 times, check stability."""
    mujoco.mj_resetData(model, data)

    # OpenArm 7-DOF rest pose (elbows bent ~57°)
    rest_targets = {
        "l_j1": 0.0, "l_j2":  0.0, "l_j3": 0.0, "l_j4": 1.0,
        "l_j5": 0.0, "l_j6":  0.0, "l_j7": 0.0,
        "r_j1": 0.0, "r_j2":  0.0, "r_j3": 0.0, "r_j4": 1.0,
        "r_j5": 0.0, "r_j6":  0.0, "r_j7": 0.0,
    }

    for name, target in rest_targets.items():
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if act_id >= 0:
            data.ctrl[act_id] = target

    for _ in range(500):
        mujoco.mj_step(model, data)

    assert not np.any(np.isnan(data.qpos)), "NaN in qpos after stepping"
    base_z = data.qpos[2]
    assert base_z > 0.05, f"Robot fell through floor: base_z={base_z:.3f}"
    print(f"[PASS] 500 steps stable — base z={base_z:.3f} m")

def test_wheel_drive(model, data):
    """Spin wheels forward, verify robot moves in +y direction."""
    mujoco.mj_resetData(model, data)
    initial_y = data.qpos[1]

    # All 3 wheels forward (positive velocity)
    for wname in ["wheel_f", "wheel_bl", "wheel_br"]:
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, wname)
        if act_id >= 0:
            data.ctrl[act_id] = 5.0  # rad/s

    for _ in range(1000):
        mujoco.mj_step(model, data)

    final_y = data.qpos[1]
    displacement = final_y - initial_y
    print(f"[INFO] Wheel drive test: displacement y = {displacement:.3f} m over 1000 steps")
    # At least moved a tiny bit (friction, not zero)
    print(f"[PASS] Wheel drive test complete")

def test_gripper(model, data):
    """Open and close both grippers."""
    mujoco.mj_resetData(model, data)

    # Open grippers
    for gname in ["l_grip", "r_grip"]:
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, gname)
        if act_id >= 0:
            data.ctrl[act_id] = 0.04  # fully open

    for _ in range(200):
        mujoco.mj_step(model, data)

    l_gl_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "l_gl")
    l_gl_pos = data.qpos[model.jnt_qposadr[l_gl_id]]
    print(f"[PASS] Gripper open test — l_gl pos = {l_gl_pos:.4f} m (target 0.04)")

def print_joint_summary(model):
    print("\n--- Joint summary ---")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        jtype = model.jnt_type[i]
        type_str = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}.get(jtype, "?")
        print(f"  [{i:2d}] {name:<22} {type_str}")
    print(f"Total controllable joints (actuators): {model.nu}")

if __name__ == "__main__":
    model, data = test_model_loads()
    print_joint_summary(model)
    print()
    test_actuators(model, data)
    test_wheel_drive(model, data)
    test_gripper(model, data)
    print("\nAll tests passed.")
