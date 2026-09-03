import mujoco

model = mujoco.MjModel.from_xml_path("/Users/dishaswamy/Desktop/Qudraped/ugpquadruped/quadruped/urdf/leg.urdf")
mujoco.mj_saveLastXML("leg_generated.xml", model)