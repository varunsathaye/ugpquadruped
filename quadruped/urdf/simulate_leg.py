import mujoco
import mujoco.viewer
import numpy as np
import time


# ============================================================
# LOAD MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path("leg.urdf")
data = mujoco.MjData(model)


# ============================================================
# JOINTS
# ============================================================

j1 = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "rectangle_to_link1"
)

j2 = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "link1_to_link2"
)


if j1 == -1:
    raise RuntimeError("rectangle_to_link1 not found")

if j2 == -1:
    raise RuntimeError("link1_to_link2 not found")


q1 = model.jnt_qposadr[j1]
q2 = model.jnt_qposadr[j2]


# ============================================================
# LEG GEOMETRY
# ============================================================

# Link 1 hinge-to-hinge distance
# Measured precisely from link1.STL (circle fit on hole walls):
# J1=(10.0,130.0)mm, J2=(10.0,10.0)mm -> 120.0mm exactly
L1 = 0.120

# Link 2 length: distance from the 2nd hole (new attachment point,
# per the parallel-mechanism equivalence) to the foot sphere center.
# Hole2=(16.0,136.0)mm, foot=(16.0,16.0,16.0)mm -> 120.0mm exactly
# (equal to L1, as expected for the 2nd-hole attachment)
L2 = 0.120


# ============================================================
# WALKING PARAMETERS
# ============================================================

STEP_LENGTH = 0.12       # metres
STEP_HEIGHT = 0.045      # metres

GAIT_FREQUENCY = 0.5     # walking cycles/sec

# Forward direction:
# +X = forward
FORWARD_SIGN = 1.0


# ============================================================
# INITIAL FOOT POSITION
# ============================================================

# These values are deliberately chosen relative to J1.
#
# You will tune these after seeing the first gait.

X_CENTER = 0.00
Y_GROUND = -0.20


# ============================================================
# INVERSE KINEMATICS
# ============================================================

def inverse_kinematics(x, y):
    """
    2-link planar inverse kinematics.

    J1 is at (0,0).
    x = forward direction
    y = downward/upward leg direction.

    Returns:
        q1, q2
    """

    r2 = x*x + y*y

    cos_q2 = (
        r2 - L1*L1 - L2*L2
    ) / (2.0 * L1 * L2)

    # Numerical protection
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)

    # Elbow configuration
    sin_q2 = np.sqrt(
        max(0.0, 1.0 - cos_q2*cos_q2)
    )

    q2 = np.arctan2(
        sin_q2,
        cos_q2
    )

    q1 = (
        np.arctan2(y, x)
        -
        np.arctan2(
            L2 * np.sin(q2),
            L1 + L2 * np.cos(q2)
        )
    )

    return q1, q2


# ============================================================
# FOOT TRAJECTORY
# ============================================================

def foot_trajectory(t):
    """
    Walking cycle:

        0 -> 0.5 : STANCE
        0.5 -> 1 : SWING

    During stance:
        foot moves backward relative to body.

    During swing:
        foot lifts and moves forward.
    """

    phase = (
        t * GAIT_FREQUENCY
    ) % 1.0


    # --------------------------------------------------------
    # STANCE
    # --------------------------------------------------------

    if phase < 0.5:

        s = phase / 0.5

        # Foot moves backward relative to body
        x = (
            X_CENTER
            +
            FORWARD_SIGN
            *
            (
                STEP_LENGTH / 2
                -
                STEP_LENGTH * s
            )
        )

        y = Y_GROUND

        return x, y


    # --------------------------------------------------------
    # SWING
    # --------------------------------------------------------

    else:

        s = (phase - 0.5) / 0.5

        # Forward movement
        x = (
            X_CENTER
            +
            FORWARD_SIGN
            *
            (
                -STEP_LENGTH / 2
                +
                STEP_LENGTH * s
            )
        )

        # Smooth foot lift
        y = (
            Y_GROUND
            +
            STEP_HEIGHT
            *
            np.sin(np.pi * s)
        )

        return x, y


# ============================================================
# INITIALIZE
# ============================================================

x0, y0 = foot_trajectory(0.0)

q1_initial, q2_initial = inverse_kinematics(
    x0,
    y0
)

data.qpos[q1] = q1_initial
data.qpos[q2] = q2_initial

mujoco.mj_forward(model, data)


# ============================================================
# PRINT INFORMATION
# ============================================================

print()
print("==============================================")
print("        QUADRUPED LEG GAIT TEST")
print("==============================================")
print()
print("J1:", q1)
print("J2:", q2)
print()
print("Link 1:", L1, "m")
print("Link 2:", L2, "m")
print()
print("Step length:", STEP_LENGTH, "m")
print("Step height:", STEP_HEIGHT, "m")
print("Frequency:", GAIT_FREQUENCY, "Hz")
print()
print("Forward direction: +X")
print()


# ============================================================
# SIMULATION
# ============================================================

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    start_time = time.time()

    while viewer.is_running():

        # Current simulation time
        t = time.time() - start_time


        # ----------------------------------------------------
        # Desired FOOT position
        # ----------------------------------------------------

        x_des, y_des = foot_trajectory(t)


        # ----------------------------------------------------
        # Convert foot position → joint angles
        # ----------------------------------------------------

        q1_des, q2_des = inverse_kinematics(
            x_des,
            y_des
        )


        # ----------------------------------------------------
        # Apply joint angles
        # ----------------------------------------------------

        data.qpos[q1] = q1_des
        data.qpos[q2] = q2_des


        # ----------------------------------------------------
        # Update kinematics
        # ----------------------------------------------------

        mujoco.mj_forward(
            model,
            data
        )


        # ----------------------------------------------------
        # Update viewer
        # ----------------------------------------------------

        viewer.sync()

        time.sleep(0.005)
