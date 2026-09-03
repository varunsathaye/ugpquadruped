import mujoco
import mujoco.viewer
import numpy as np
import time

from leg_config import LEGS


# ============================================================
# LOAD MODEL
# Now the generated MJCF (quadruped_generated.xml) instead of URDF --
# switched so each leg could carry a real foot_site + touch sensor,
# same pattern as leg_generated.xml / python_floor.py.
# ============================================================

model = mujoco.MjModel.from_xml_path("quadruped_generated.xml")
data = mujoco.MjData(model)


# ============================================================
# WHY THE IK STILL WORKS THIS WAY
#
# Same numerically-validated approach as before: no closed-form
# formula (one was tried, was quietly wrong for this mechanism, and
# produced silently-incorrect knee bends -- see project history).
# Instead, a dense (q1,q2) grid is swept at startup for one leg of
# each type (unmirrored, mirrored), the ACTUAL foot position is read
# back via data.site_xpos (now a real named site instead of the
# geom-transform workaround used before this had sites at all), and
# that table becomes a starting guess for a few Newton-refinement
# steps per frame.
#
# One addition specific to this mechanism: for a given foot target
# there are TWO valid (q1,q2) solutions -- two different knee bends
# that land the foot in the same place -- and only one of them (q2
# constrained >= 0) matches the originally-validated single-leg
# convention. The table is built with q2 restricted to that range so
# the solver can never wander onto the other, unvalidated branch.
# ============================================================


def _find_site(name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid == -1:
        raise RuntimeError(f"site {name} not found")
    return sid


def build_fk_table(leg_name, n=120, qlim=1.55):
    j1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"rectangle_to_link1_{leg_name}")
    j2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"link1_to_link2_{leg_name}")
    site_id = _find_site(f"foot_site_{leg_name}")
    mujoco.mj_forward(model, data)
    hip = data.xanchor[j1].copy()

    q1_range = np.linspace(-qlim, qlim, n)
    q2_range = np.linspace(0.0, qlim, n)  # validated branch only, see note above
    table = np.zeros((n, n, 2))
    for i, q1v in enumerate(q1_range):
        for k, q2v in enumerate(q2_range):
            data.qpos[model.jnt_qposadr[j1]] = q1v
            data.qpos[model.jnt_qposadr[j2]] = q2v
            mujoco.mj_forward(model, data)
            fw = data.site_xpos[site_id]
            table[i, k, 0] = fw[0] - hip[0]
            table[i, k, 1] = fw[2] - hip[2]
    return j1, j2, site_id, hip, q1_range, q2_range, table


print("Building leg kinematics tables (one-time, ~3-6s)...")
_UNMIRRORED_REF = build_fk_table("FR")
_MIRRORED_REF = build_fk_table("FL")
print("Done.")


def _nearest_guess(ref, tx, tz):
    _, _, _, _, q1_range, q2_range, table = ref
    d2 = (table[:, :, 0] - tx) ** 2 + (table[:, :, 1] - tz) ** 2
    i, k = np.unravel_index(np.argmin(d2), d2.shape)
    return q1_range[i], q2_range[k]


def solve_leg_ik(leg_j1, leg_j2, leg_site, leg_hip, ref, tx, tz, warm_start=None, iters=4, h=1e-4):
    def foot_xz(q1v, q2v):
        data.qpos[model.jnt_qposadr[leg_j1]] = q1v
        data.qpos[model.jnt_qposadr[leg_j2]] = q2v
        mujoco.mj_forward(model, data)
        fw = data.site_xpos[leg_site]
        return np.array([fw[0] - leg_hip[0], fw[2] - leg_hip[2]])

    q1v, q2v = warm_start if warm_start is not None else _nearest_guess(ref, tx, tz)
    target = np.array([tx, tz])
    for _ in range(iters):
        f = foot_xz(q1v, q2v) - target
        f1 = (foot_xz(q1v + h, q2v) - foot_xz(q1v - h, q2v)) / (2 * h)
        f2 = (foot_xz(q1v, q2v + h) - foot_xz(q1v, q2v - h)) / (2 * h)
        J = np.column_stack([f1, f2])
        try:
            delta = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            break
        q1v = float(np.clip(q1v + delta[0], -1.55, 1.55))
        q2v = float(np.clip(q2v + delta[1], 0.0, 1.55))  # stay on the validated branch
    return q1v, q2v


# ============================================================
# PER-LEG SETUP
# ============================================================

LEG_PHASE = {"FR": 0.0, "BL": 0.0, "FL": 0.5, "BR": 0.5}  # trot: diagonal pairs

leg_info = {}
for name in LEG_PHASE:
    j1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"rectangle_to_link1_{name}")
    j2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"link1_to_link2_{name}")
    site_id = _find_site(f"foot_site_{name}")
    sensor_adr = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"foot_force_{name}")]
    mujoco.mj_forward(model, data)
    hip = data.xanchor[j1].copy()
    ref = _MIRRORED_REF if LEGS[name]["mirror"] else _UNMIRRORED_REF
    leg_info[name] = dict(j1=j1, j2=j2, site=site_id, sensor_adr=sensor_adr, hip=hip, ref=ref, warm_start=None)


# ============================================================
# GAIT PARAMETERS
# Re-verified against the branch-constrained workspace (see project
# history for how these were measured, with margin, from the actual
# reachable Z range at each X across the stride).
# ============================================================

STEP_LENGTH = 0.12
STANCE_Z = -0.184
STEP_HEIGHT = 0.012
GAIT_FREQUENCY = 0.5
X_CENTER = 0.0


def foot_trajectory(t, phase_offset):
    phase = (t * GAIT_FREQUENCY + phase_offset) % 1.0
    if phase < 0.5:
        s = phase / 0.5
        x = X_CENTER + (STEP_LENGTH / 2 - STEP_LENGTH * s)
        z = STANCE_Z
    else:
        s = (phase - 0.5) / 0.5
        x = X_CENTER + (-STEP_LENGTH / 2 + STEP_LENGTH * s)
        z = STANCE_Z + STEP_HEIGHT * np.sin(np.pi * s)
    return x, z


# ============================================================
# INITIALIZE
# ============================================================

for name, info in leg_info.items():
    x0, z0 = foot_trajectory(0.0, LEG_PHASE[name])
    q1v, q2v = solve_leg_ik(info["j1"], info["j2"], info["site"], info["hip"], info["ref"], x0, z0)
    data.qpos[model.jnt_qposadr[info["j1"]]] = q1v
    data.qpos[model.jnt_qposadr[info["j2"]]] = q2v
    info["warm_start"] = (q1v, q2v)

mujoco.mj_forward(model, data)

print()
print("==============================================")
print("        QUADRUPED TROT GAIT + FOOT SENSORS")
print("==============================================")
print("Legs:", list(leg_info))
print("Stance z (rel. to hip):", STANCE_Z, "  Swing lift:", STEP_HEIGHT)
print()


# ============================================================
# CAMERA PRESETS
# ============================================================

LOOKAT = [0.15, 0.0, 0.01]
VIEWS = {
    ord("R"): dict(azimuth=135, elevation=-20, distance=0.95),
    ord("F"): dict(azimuth=90, elevation=-12, distance=0.9),
    ord("B"): dict(azimuth=270, elevation=-12, distance=0.9),
    ord("T"): dict(azimuth=90, elevation=-89, distance=0.95),
    ord("S"): dict(azimuth=0, elevation=-12, distance=0.9),
}
VIEW_NAMES = {ord("R"): "reset/default", ord("F"): "front", ord("B"): "back",
              ord("T"): "top", ord("S"): "side"}


def apply_view(cam, view):
    cam.azimuth = view["azimuth"]
    cam.elevation = view["elevation"]
    cam.distance = view["distance"]
    cam.lookat[:] = LOOKAT


_cam_ref = []


def key_callback(keycode):
    view = VIEWS.get(keycode)
    if view is not None and _cam_ref:
        apply_view(_cam_ref[0], view)
        print(f"[camera] switched to {VIEW_NAMES[keycode]} view")


# ============================================================
# SIMULATION
# Each frame also reads the touch sensors (data.sensordata at each
# leg's sensor_adr) so ground contact is visible in the console, not
# just assumed from the gait phase.
# ============================================================

PRINT_EVERY = 40  # throttle sensor printing so the console stays readable
_frame_count = 0

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    _cam_ref.append(viewer.cam)
    apply_view(viewer.cam, VIEWS[ord("R")])

    start_time = time.time()

    while viewer.is_running():
        t = time.time() - start_time

        for name, info in leg_info.items():
            x_des, z_des = foot_trajectory(t, LEG_PHASE[name])
            q1v, q2v = solve_leg_ik(
                info["j1"], info["j2"], info["site"], info["hip"], info["ref"],
                x_des, z_des, warm_start=info["warm_start"], iters=3,
            )
            info["warm_start"] = (q1v, q2v)
            data.qpos[model.jnt_qposadr[info["j1"]]] = q1v
            data.qpos[model.jnt_qposadr[info["j2"]]] = q2v

        mujoco.mj_forward(model, data)

        _frame_count += 1
        if _frame_count % PRINT_EVERY == 0:
            forces = {name: round(float(data.sensordata[info["sensor_adr"]]), 3)
                      for name, info in leg_info.items()}
            print(f"t={t:5.2f}  foot_force:", forces)

        viewer.sync()
        time.sleep(0.005)
