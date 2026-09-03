import mujoco
import numpy as np
import os

# ============================================================
# LOAD MODEL — the generated MJCF, now with foot_site added
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(SCRIPT_DIR, "leg_generated.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
print("loaded model from:", XML_PATH)
print("number of sites in model:", model.nsite)
for i in range(model.nsite):
    print(" site:", model.site(i).name)

data = mujoco.MjData(model)

j1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rectangle_to_link1")
j2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "link1_to_link2")
q1 = model.jnt_qposadr[j1]
q2 = model.jnt_qposadr[j2]

foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot_site")
if foot_id == -1:
    raise RuntimeError("foot_site not found — check it was added inside <body name=\"link2\"> in leg_generated.xml")

# ============================================================
# SAME GEOMETRY / IK / TRAJECTORY AS YOUR MAIN SCRIPT
# ============================================================
L1 = 0.120
L2 = 0.120
STEP_LENGTH = 0.12
STEP_HEIGHT = 0.045
GAIT_FREQUENCY = 0.5
FORWARD_SIGN = 1.0
X_CENTER = 0.00
Y_GROUND = -0.20

def inverse_kinematics(x, y):
    r2 = x*x + y*y
    cos_q2 = (r2 - L1*L1 - L2*L2) / (2.0*L1*L2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    sin_q2 = np.sqrt(max(0.0, 1.0 - cos_q2*cos_q2))
    q2_ = np.arctan2(sin_q2, cos_q2)
    q1_ = np.arctan2(y, x) - np.arctan2(L2*np.sin(q2_), L1 + L2*np.cos(q2_))
    return q1_, q2_

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
        y = Y_GROUND + STEP_HEIGHT * np.sin(np.pi*s)
        return x, y

# ============================================================
# SAMPLE ONE FULL GAIT CYCLE — no viewer, just console output
# ============================================================
N = 100
period = 1.0 / GAIT_FREQUENCY

for i in range(N):
    t = i / N * period
    x, y = foot_trajectory(t)
    q1_des, q2_des = inverse_kinematics(x, y)

    data.qpos[q1] = q1_des
    data.qpos[q2] = q2_des
    mujoco.mj_forward(model, data)

    phase = "STANCE" if (t * GAIT_FREQUENCY) % 1.0 < 0.5 else "SWING "
    print(f"{phase}  world foot pos: {data.site_xpos[foot_id]}")