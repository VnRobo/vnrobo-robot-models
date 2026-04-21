"""Launch MuJoCo passive viewer for VNR-WH1.

Usage:
    python examples/view_model.py              # robot only
    python examples/view_model.py --scene      # with manipulation objects
"""

import argparse
import os
import mujoco
import mujoco.viewer

ROOT = os.path.dirname(os.path.dirname(__file__))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", action="store_true", help="Load default scene")
    args = parser.parse_args()

    if args.scene:
        xml_path = os.path.join(ROOT, "vnr_wh1", "scenes", "default.xml")
    else:
        xml_path = os.path.join(ROOT, "vnr_wh1", "vnr_wh1.xml")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    print(f"VNR-WH1 loaded")
    print(f"  nq (generalized positions): {model.nq}")
    print(f"  nv (generalized velocities): {model.nv}")
    print(f"  nu (actuators): {model.nu}")
    print(f"  nbody: {model.nbody}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetData(model, data)
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
