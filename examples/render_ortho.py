"""Render VNR-WH1 as orthographic 2D views: front, back, left, right, top.
Used for structural review — shows silhouette + visible gaps clearly.

Forward convention: robot forward = world -X (body_link0 has 90deg-Z quat).
Output: /tmp/vnr_wh1_screenshots/ortho_{view}.png
"""
import os
import mujoco
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(__file__))
XML  = os.path.join(ROOT, "vnr_wh1", "vnr_wh1.xml")

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
mujoco.mj_resetData(m, d)
for _ in range(400): mujoco.mj_step(m, d)

renderer = mujoco.Renderer(m, height=900, width=600)

# Near-ortho: far distance + small FOV. At 8m with fov=15deg, perspective negligible.
# (az, el, dist, lookat_z, label, fov)
views = [
    ("front",  180,   0, 8.0, 0.75, "FRONT  (camera at world -X)"),
    ("back",     0,   0, 8.0, 0.75, "BACK   (camera at world +X)"),
    ("left",    90,   0, 8.0, 0.75, "LEFT   (camera at world +Y)"),
    ("right", -90,   0, 8.0, 0.75, "RIGHT  (camera at world -Y)"),
    ("top",      0, -89, 8.0, 0.00, "TOP    (camera at world +Z)"),
]

# Shrink fov so far camera acts nearly orthographic
m.vis.global_.fovy = 15.0

out_dir = "/tmp/vnr_wh1_screenshots"
os.makedirs(out_dir, exist_ok=True)

for name, az, el, dist, lookz, label in views:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 0, lookz]
    cam.distance  = dist
    cam.azimuth   = az
    cam.elevation = el
    renderer.update_scene(d, camera=cam)
    img = renderer.render()

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    # Top-left label
    draw.rectangle([(8, 8), (8 + 6.5 * len(label) + 8, 36)], fill=(0, 0, 0, 200))
    draw.text((14, 14), label, fill=(255, 255, 255))
    path = os.path.join(out_dir, f"ortho_{name}.png")
    pil.save(path)
    print(f"Saved {path}")

renderer.close()
print("\nOpen the PNGs side-by-side to review structural silhouette.")
