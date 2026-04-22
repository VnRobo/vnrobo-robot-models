"""VNR-WH1 design invariants — Playwright-equivalent automated review.

Validates v0.9 design contract after every edit:
  - Gripper world z clears base top by >= 0.30 m at rest pose.
  - Shoulder world z lies within 1.20..1.30 m (workstation humanoid spec).
  - OpenArm native body_link0 bridge meshes (body_v1, v4, v5) are visible
    (group != 4). These carry the shoulder bridge + camera bay.
  - head_cam site + head_cam_housing + head_cam_lens geoms exist.
  - v0.9 NEW: torso_column + torso_band exist and fill the shoulder gap —
    centerline corridor |x|<0.06, |y|<0.06 has at least one geom at every
    5 cm slice from world z=0.30 to z=1.20 (no floating shoulders).
  - v0.9 NEW: head_cam faces perpendicular to the arm-span axis (so the
    camera does not look along the arm-span direction).
  - No self-collisions at rest (only wheel<->floor contacts allowed).

Run:  python3 examples/visual_test.py
Exit: 0 on pass, non-zero with diagnostic on fail.
"""
import os, sys, numpy as np, mujoco

ROOT = os.path.dirname(os.path.dirname(__file__))
XML  = os.path.join(ROOT, "vnr_wh1", "vnr_wh1.xml")

BASE_TOP_Z      = 0.30
GRIPPER_MIN_CLR = 0.30
SHOULDER_RANGE  = (1.20, 1.30)
VISIBLE_BRIDGE  = ("body_v1", "body_v4", "body_v5")
HIDDEN_INTERNAL = ("body_v0", "body_v2", "body_v3")
CAM_GEOMS       = ("head_cam_housing", "head_cam_lens")
CAM_SITE        = "head_cam"

def fail(msg):
    print(f"FAIL  {msg}"); sys.exit(1)

def ok(msg):
    print(f"PASS  {msg}")

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m); mujoco.mj_forward(m, d)

def body_z(name):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0: fail(f"body missing: {name}")
    return float(d.xpos[bid, 2])

def geom_group(name):
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    return None if gid < 0 else int(m.geom_group[gid])

# 1. Kinematic heights
sh_l = body_z("openarm_left_link0")
sh_r = body_z("openarm_right_link0")
if not (SHOULDER_RANGE[0] <= sh_l <= SHOULDER_RANGE[1]):
    fail(f"left shoulder z={sh_l:.3f} outside {SHOULDER_RANGE}")
if not (SHOULDER_RANGE[0] <= sh_r <= SHOULDER_RANGE[1]):
    fail(f"right shoulder z={sh_r:.3f} outside {SHOULDER_RANGE}")
ok(f"shoulders within spec: L={sh_l:.3f}  R={sh_r:.3f}")

hand_l = body_z("openarm_left_hand");  hand_r = body_z("openarm_right_hand")
clr_l = hand_l - BASE_TOP_Z; clr_r = hand_r - BASE_TOP_Z
if min(clr_l, clr_r) < GRIPPER_MIN_CLR:
    fail(f"gripper clearance too small: L={clr_l:.3f} R={clr_r:.3f} (need >= {GRIPPER_MIN_CLR})")
ok(f"gripper clearance: L={clr_l:.3f}m  R={clr_r:.3f}m  (base_top={BASE_TOP_Z})")

# 2. Visibility contract on body_link0 meshes (geom name = mesh name by MuJoCo
# default when not renamed). Iterate over all geoms looking for mesh refs.
vis_seen  = set(); hid_seen = set()
for gid in range(m.ngeom):
    if m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH: continue
    mesh_id = m.geom_dataid[gid]
    if mesh_id < 0: continue
    mesh_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
    grp = int(m.geom_group[gid])
    if mesh_name in VISIBLE_BRIDGE:
        if grp == 4: fail(f"bridge mesh {mesh_name} is hidden (group=4)")
        vis_seen.add(mesh_name)
    if mesh_name in HIDDEN_INTERNAL:
        hid_seen.add((mesh_name, grp))

missing = set(VISIBLE_BRIDGE) - vis_seen
if missing: fail(f"bridge meshes missing from model: {missing}")
ok(f"bridge meshes visible: {sorted(vis_seen)}")

for name, grp in hid_seen:
    if grp != 4: fail(f"internal mesh {name} should be hidden (group=4) but is group={grp}")
ok(f"internal meshes hidden: {sorted(n for n,_ in hid_seen)}")

# 3. Camera hardware present
for gname in CAM_GEOMS:
    if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname) < 0:
        fail(f"camera geom missing: {gname}")
ok(f"camera geoms present: {CAM_GEOMS}")
if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, CAM_SITE) < 0:
    fail(f"camera site missing: {CAM_SITE}")
ok(f"camera site present: {CAM_SITE}")

# 4. Silhouette continuity — no empty column between base and shoulders.
#    Ray-sample vertical slices at world z = 0.30..1.20 every 5 cm. At each z,
#    verify at least one geom (visual class, group != 4) is present in the
#    centerline corridor |x|<0.06, |y|<0.06. Catches "floating shoulders".
def sample_corridor_hit(z):
    # For every visual/collision mesh/box geom owned by base or openarm_body_link0
    # sub-tree, check if any has world aabb containing a point in corridor.
    # Simplification: test box/cylinder geoms that are axis-aligned to world z
    # (our added primitives are); skip meshes (we trust body_link0 meshes by
    # visibility contract already verified above).
    for gid in range(m.ngeom):
        if int(m.geom_group[gid]) == 4: continue
        if m.geom_type[gid] not in (mujoco.mjtGeom.mjGEOM_BOX,
                                    mujoco.mjtGeom.mjGEOM_CYLINDER): continue
        px, py, pz = d.geom_xpos[gid]
        sx, sy, sz = m.geom_size[gid]
        if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            # cylinder size = (radius, half_height, _)
            if abs(pz - z) > sy: continue
            if (px ** 2 + py ** 2) > sx ** 2: continue
            # check corridor overlap: disc of radius sx centered at (px,py)
            dx = max(0.0, abs(px) - 0.06)
            dy = max(0.0, abs(py) - 0.06)
            if dx * dx + dy * dy <= sx * sx:
                return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid)
        else:
            # axis-aligned box
            if abs(pz - z) > sz: continue
            if abs(px) > sx + 0.06 or abs(py) > sy + 0.06: continue
            return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid)
    return None

missing_slices = []
z_vals = [round(0.30 + 0.05 * i, 2) for i in range(int((1.20 - 0.30) / 0.05) + 1)]
for z in z_vals:
    hit = sample_corridor_hit(z)
    if hit is None:
        missing_slices.append(z)
if missing_slices:
    fail(f"silhouette gap: no geom in centerline corridor at world z = {missing_slices}")
ok(f"silhouette continuous z=0.30..1.20 ({len(z_vals)} slices, corridor |x|,|y|<0.06)")

# 5. Camera orientation — head_cam must not look along the arm-span axis.
#    In our model arm-span is world x (shoulders at world x=±0.031). The
#    head_cam housing cylinder's world axis determines the camera direction.
cam_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "head_cam_housing")
if cam_gid < 0: fail("head_cam_housing missing")
cam_axis_world = d.geom_xmat[cam_gid].reshape(3, 3) @ np.array([0, 0, 1.0])  # local +z = cyl axis
cam_axis_world /= np.linalg.norm(cam_axis_world) + 1e-9
if abs(cam_axis_world[0]) > abs(cam_axis_world[1]):
    fail(f"head_cam axis mostly along world X (arm-span): {cam_axis_world}")
ok(f"head_cam axis perpendicular to arm-span: world={cam_axis_world.round(3).tolist()}")

# 6. No self-collisions at rest — only wheel<->floor should touch.
for _ in range(3): mujoco.mj_step(m, d)
floor_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
bad = []
for i in range(d.ncon):
    c = d.contact[i]
    if floor_gid in (c.geom1, c.geom2): continue
    g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
    g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
    bad.append((g1, g2, c.dist))
if bad: fail(f"self-collisions at rest: {bad}")
ok(f"no self-collisions at rest ({d.ncon} floor contacts only)")

print("\nAll invariants hold. v0.9 design contract OK.")
