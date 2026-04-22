"""VNR-WH1 design invariants — Playwright-equivalent automated review.

Validates the v1.1 design contract after every edit:
  - Gripper world z clears base top by >= 0.30 m at rest pose.
  - Shoulder world z lies within 1.20..1.30 m (workstation humanoid spec).
  - VISIBLE: body_v5 (native OpenArm shoulder-bridge yoke) MUST be visible
    — it is the branded joint connector between the two arm mounts. Its
    world aabb center must sit at shoulder height (z > 1.10).
  - HIDDEN: body_v0..v4 MUST be hidden (group=4) — hip/waist fasteners
    and internal backbone, superseded by torso_column.
  - torso_column + torso_band geoms exist (slim pillar + brand band).
  - head_cam site + head_cam_housing + head_cam_lens geoms exist.
  - head_cam anchoring: head_cam_housing world z in [1.15, 1.30] AND
    within 0.05 m of body_v5 world aabb (catches the v1.0 regression
    where head_cam was stuck at hip level, floating free on the side).
  - Silhouette continuity: centerline corridor |x|<0.06, |y|<0.06 has
    at least one geom at every 5 cm slice from world z=0.30 to z=1.20.
  - head_cam faces perpendicular to the arm-span axis.
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
HIDDEN_NATIVE   = ("body_v0", "body_v1", "body_v2", "body_v3", "body_v4")
VISIBLE_BRIDGE  = "body_v5"
REQUIRED_BOXES  = ("torso_column", "torso_band")
CAM_GEOMS       = ("head_cam_housing", "head_cam_lens")
CAM_SITE        = "head_cam"
CAM_Z_RANGE     = (1.15, 1.30)    # head cam must be at head level
CAM_BRIDGE_SLOP = 0.05            # max distance from body_v5 aabb

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

# 2. Visibility contract: v0..v4 hidden; v5 visible and at shoulder height.
bridge_gid, bridge_aabb = None, None
hidden_seen = set()
for gid in range(m.ngeom):
    if m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH: continue
    mesh_id = m.geom_dataid[gid]
    if mesh_id < 0: continue
    mesh_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
    grp = int(m.geom_group[gid])
    if mesh_name in HIDDEN_NATIVE:
        if grp != 4:
            fail(f"native hip mesh {mesh_name} must be hidden (group=4) but is group={grp}")
        hidden_seen.add(mesh_name)
    elif mesh_name == VISIBLE_BRIDGE:
        if grp == 4:
            fail(f"{VISIBLE_BRIDGE} must be VISIBLE but is hidden (group=4)")
        bridge_gid = gid
missing_hidden = set(HIDDEN_NATIVE) - hidden_seen
if missing_hidden: fail(f"native hip meshes missing from model: {missing_hidden}")
if bridge_gid is None: fail(f"{VISIBLE_BRIDGE} geom not found")
ok(f"native hip meshes hidden: {sorted(hidden_seen)}")

# compute body_v5 world aabb from its mesh verts + geom transform
bmesh_id = m.geom_dataid[bridge_gid]
va = m.mesh_vertadr[bmesh_id]; vn = m.mesh_vertnum[bmesh_id]
verts_local = m.mesh_vert[va:va+vn]                          # (N,3) in mesh space
xmat = d.geom_xmat[bridge_gid].reshape(3, 3)
xpos = d.geom_xpos[bridge_gid]
verts_world = (xmat @ verts_local.T).T + xpos                # (N,3)
bridge_aabb = (verts_world.min(axis=0), verts_world.max(axis=0))
bridge_center_z = 0.5 * (bridge_aabb[0][2] + bridge_aabb[1][2])
if bridge_center_z < 1.10:
    fail(f"{VISIBLE_BRIDGE} center z={bridge_center_z:.3f} is not at shoulder level (>=1.10)")
ok(f"{VISIBLE_BRIDGE} visible at shoulder: center_z={bridge_center_z:.3f} "
   f"aabb_y=[{bridge_aabb[0][1]:+.3f},{bridge_aabb[1][1]:+.3f}]")

for gname in REQUIRED_BOXES:
    if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname) < 0:
        fail(f"required primitive missing: {gname}")
ok(f"primitives present: {REQUIRED_BOXES}")

# 3. Camera hardware present
for gname in CAM_GEOMS:
    if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname) < 0:
        fail(f"camera geom missing: {gname}")
ok(f"camera geoms present: {CAM_GEOMS}")
if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, CAM_SITE) < 0:
    fail(f"camera site missing: {CAM_SITE}")
ok(f"camera site present: {CAM_SITE}")

# 3b. head_cam must be at head level AND anchored to body_v5 (catches
#     "floating camera on the side profile" regression from v1.0).
cam_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "head_cam_housing")
cam_world = d.geom_xpos[cam_gid]
if not (CAM_Z_RANGE[0] <= cam_world[2] <= CAM_Z_RANGE[1]):
    fail(f"head_cam_housing world z={cam_world[2]:.3f} outside head band {CAM_Z_RANGE} "
         f"— looks like it is anchored at hip level")
lo, hi = bridge_aabb
dx = max(lo[0]-cam_world[0], 0.0, cam_world[0]-hi[0])
dy = max(lo[1]-cam_world[1], 0.0, cam_world[1]-hi[1])
dz = max(lo[2]-cam_world[2], 0.0, cam_world[2]-hi[2])
dist = float(np.sqrt(dx*dx + dy*dy + dz*dz))
if dist > CAM_BRIDGE_SLOP:
    fail(f"head_cam_housing floats {dist*1000:.0f}mm away from {VISIBLE_BRIDGE} aabb "
         f"(slop={CAM_BRIDGE_SLOP*1000:.0f}mm). Camera will appear disconnected in side views.")
ok(f"head_cam anchored to {VISIBLE_BRIDGE}: world_z={cam_world[2]:.3f} "
   f"aabb_dist={dist*1000:.0f}mm")

# 4. Silhouette continuity — no empty column between base and shoulders.
#    Ray-sample vertical slices at world z = 0.30..1.20 every 5 cm. At each z,
#    verify at least one axis-aligned box/cylinder geom (visual, group != 4)
#    covers the centerline corridor |x|<0.06, |y|<0.06.
def sample_corridor_hit(z):
    for gid in range(m.ngeom):
        if int(m.geom_group[gid]) == 4: continue
        if m.geom_type[gid] not in (mujoco.mjtGeom.mjGEOM_BOX,
                                    mujoco.mjtGeom.mjGEOM_CYLINDER): continue
        px, py, pz = d.geom_xpos[gid]
        sx, sy, sz = m.geom_size[gid]
        if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            if abs(pz - z) > sy: continue
            if (px ** 2 + py ** 2) > sx ** 2: continue
            dx = max(0.0, abs(px) - 0.06)
            dy = max(0.0, abs(py) - 0.06)
            if dx * dx + dy * dy <= sx * sx:
                return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid)
        else:
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

# 5. Camera orientation — head_cam must not look along the arm-span axis (world X).
cam_axis_world = d.geom_xmat[cam_gid].reshape(3, 3) @ np.array([0, 0, 1.0])
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

print("\nAll invariants hold. v1.1 design contract OK.")
