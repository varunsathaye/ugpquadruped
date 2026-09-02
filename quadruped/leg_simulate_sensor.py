import mujoco
import mujoco.viewer
import numpy as np
import time
import os

# ============================================================
# LOAD MODEL — now leg_generated.xml, with foot_site, the
# touch sensor, and the floor all baked in
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(SCRIPT_DIR, "/Users/dishaswamy/Desktop/Qudraped/ugpquadruped/quadruped/urdf/leg_generated.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)


# ============================================================
# JOINTS
# ============================================================

j1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rectangle_to_link1")
j2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "link1_to_link2")

if j1 == -1:
    raise RuntimeError("rectangle_to_link1 not found")
if j2 == -1:
    raise RuntimeError("link1_to_link2 not found")

q1 = model.jnt_qposadr[j1]
q2 = model.jnt_qposadr[j2]


# ============================================================
# FOOT FORCE SENSOR
# ============================================================

foot_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "foot_force")
if foot_sensor_id == -1:
    raise RuntimeError("foot_force sensor not found — check <sensor> block in leg_generated.xml")

foot_sensor_adr = model.sensor_adr[foot_sensor_id]


# ============================================================
# LEG GEOMETRY
# ============================================================

L1 = 0.120
L2 = 0.120

STEP_LENGTH = 0.12
STEP_HEIGHT = 0.045
GAIT_FREQUENCY = 0.5
FORWARD_SIGN = 1.0

X_CENTER = 0.00
Y_GROUND = -0.20


# ============================================================
# INVERSE KINEMATICS
# ============================================================

def inverse_kinematics(x, y):
    r2 = x*x + y*y
    cos_q2 = (r2 - L1*L1 - L2*L2) / (2.0 * L1 * L2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    sin_q2 = np.sqrt(max(0.0, 1.0 - cos_q2*cos_q2))
    q2_ = np.arctan2(sin_q2, cos_q2)
    q1_ = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2_), L1 + L2 * np.cos(q2_))
    return q1_, q2_


# ============================================================
# FOOT TRAJECTORY
# ============================================================

def foot_trajectory(t):
    phase = (t * GAIT_FREQUENCY) % 1.0

    if phase < 0.5:
        s = phase / 0.5
        x = X_CENTER + FORWARD_SIGN * (STEP_LENGTH/2 - STEP_LENGTH*s)
        y = Y_GROUND
        return x, y
    else:
        s = (phase - 0.5) / 0.5
        x = X_CENTER + FORWARD_SIGN * (-STEP_LENGTH/2 + STEP_LENGTH*s)
        y = Y_GROUND + STEP_HEIGHT * np.sin(np.pi * s)
        return x, y


# ============================================================
# INITIALIZE
# ============================================================

x0, y0 = foot_trajectory(0.0)
q1_initial, q2_initial = inverse_kinematics(x0, y0)
data.qpos[q1] = q1_initial
data.qpos[q2] = q2_initial
mujoco.mj_forward(model, data)


# ============================================================
# SIMULATION
# ============================================================

with mujoco.viewer.launch_passive(model, data) as viewer:

    start_time = time.time()

    while viewer.is_running():

        t = time.time() - start_time

        x_des, y_des = foot_trajectory(t)
        q1_des, q2_des = inverse_kinematics(x_des, y_des)

        data.qpos[q1] = q1_des
        data.qpos[q2] = q2_des

        mujoco.mj_forward(model, data)

        # -----------------------------------------------
        # READ FOOT FORCE
        # -----------------------------------------------
        foot_force = data.sensordata[foot_sensor_adr]
        print(f"foot force: {foot_force:6.2f} N")

        viewer.sync()
        time.sleep(0.005)