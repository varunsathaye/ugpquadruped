import mujoco
import mujoco.viewer
import numpy as np
import time


# ============================================================
# Load the URDF
# ============================================================

model = mujoco.MjModel.from_xml_path("leg.urdf")

data = mujoco.MjData(model)


# ============================================================
# Print information about the model
# ============================================================

print("\n========== MuJoCo Model ==========")

print("Number of bodies :", model.nbody)
print("Number of joints :", model.njnt)
print("Number of DOFs   :", model.nv)

print("\nJoints:")

for i in range(model.njnt):
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        i
    )

    print(f"  {i}: {name}")


# ============================================================
# Get joint addresses
# ============================================================

j1 = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "mild_to_rectangle"
)

j2 = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "mild_to_link1"
)

j3 = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "link1_to_link2"
)


# Check that all joints exist
if j1 == -1:
    raise RuntimeError("Joint mild_to_rectangle not found")

if j2 == -1:
    raise RuntimeError("Joint mild_to_link1 not found")

if j3 == -1:
    raise RuntimeError("Joint link1_to_link2 not found")


# ============================================================
# Find qpos locations
# ============================================================

q1 = model.jnt_qposadr[j1]
q2 = model.jnt_qposadr[j2]
q3 = model.jnt_qposadr[j3]


print("\nJoint qpos addresses:")
print("J1:", q1)
print("J2:", q2)
print("J3:", q3)


# ============================================================
# Open MuJoCo viewer
# ============================================================

with mujoco.viewer.launch_passive(model, data) as viewer:

    start_time = time.time()

    while viewer.is_running():

        t = time.time() - start_time

        # ----------------------------------------------------
        # Test movement
        #
        # These are just for checking that the joints work.
        # Remove this later when we add motors/controllers.
        # ----------------------------------------------------

        data.qpos[q1] = 0.4 * np.sin(t)
        data.qpos[q2] = 0.5 * np.sin(t)
        data.qpos[q3] = 0.6 * np.sin(t)

        # Recalculate forward kinematics
        mujoco.mj_forward(model, data)

        # Update viewer
        viewer.sync()

        time.sleep(0.01)