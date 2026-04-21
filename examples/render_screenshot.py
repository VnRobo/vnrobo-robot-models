"""Render VNR-WH1 to PNG screenshots from multiple angles."""

import os
import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(__file__))
XML = os.path.join(ROOT, "vnr_wh1", "vnr_wh1.xml")

model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

# OpenArm 7-DOF rest pose
rest = {
    "l_j1":  0.3, "l_j2":  0.0, "l_j3":  0.3, "l_j4": 1.2,
    "l_j5":  0.0, "l_j6":  0.2, "l_j7":  0.0,
    "r_j1": -0.3, "r_j2":  0.0, "r_j3": -0.3, "r_j4": 1.2,
    "r_j5":  0.0, "r_j6":  0.2, "r_j7":  0.0,
    "head_tilt": 0.1,
}
for name, val in rest.items():
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid >= 0:
        data.ctrl[aid] = val

for _ in range(400):
    mujoco.mj_step(model, data)

renderer = mujoco.Renderer(model, height=720, width=720)

# (azimuth_deg, elevation_deg, distance)
cameras = [
    ("front",    90,  -15, 2.8),
    ("side",      0,  -15, 2.8),
    ("iso",      45,  -20, 3.2),
    ("top",      90,  -80, 3.5),
]

out_dir = "/tmp/vnr_wh1_screenshots"
os.makedirs(out_dir, exist_ok=True)

for cam_name, azimuth, elevation, distance in cameras:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 0, 0.75]   # look at torso center
    cam.distance  = distance
    cam.azimuth   = azimuth
    cam.elevation = elevation

    renderer.update_scene(data, camera=cam)
    img = renderer.render()

    from PIL import Image
    path = os.path.join(out_dir, f"vnr_wh1_{cam_name}.png")
    Image.fromarray(img).save(path)
    print(f"Saved {path}")

renderer.close()
print("Done.")
