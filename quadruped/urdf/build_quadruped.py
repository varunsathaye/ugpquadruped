"""
build_quadruped.py

Generates quadruped_generated.xml (native MJCF, all 4 legs, each with a
foot_site + touch sensor) out of leg.urdf's geometry -- the same source
of truth used throughout this project -- combined with the site/sensor
pattern from leg_generated.xml (the single-leg reference with sensor
integration).

Switched from URDF to native MJCF output because sites and sensors
aren't representable in plain URDF; MJCF supports them directly, and
as a nested-body format it's a more natural fit for this generator
than URDF's separate link/joint declarations anyway.

Core idea, unchanged from before: the internal geometry of one leg
(rectangle -> link1 -> link2, foot_site, the two joint origins) never
changes between legs except for the lateral (local Z) component of
the two joint origins, which gets negated for mirrored legs -- see
mirror_origin(). Everything else differs only by the per-leg hip_pos
and the shared BASE_ROLL rotation (both from leg_config.py).

CHASSIS INTEGRATION (body1.STL):
The 4 legs used to be direct children of <worldbody>, each pinned to
its hip_pos in world space -- there was no actual robot body, just 4
independently-clamped leg rigs. This adds a real chassis body (from
body1.STL, exported from Assem2's body1 part) with a <freejoint/> so
it's a free rigid body, and nests all 4 legs inside it instead of the
world. leg_config.py's hip_pos values were already documented as
relative to a "body1 reference point," so the legs' pos= values don't
change at all -- only which element they're nested under.
"""

import math
import xml.etree.ElementTree as ET

from leg_config import LEGS, BASE_ROLL

SOURCE_URDF = "leg.urdf"
OUTPUT_XML = "quadruped_generated.xml"

# Foot-sphere center relative to link2's own body frame -- measured
# directly from link2.STL (see leg_generated.xml's foot_site, and the
# independent STL-cluster measurement earlier in this project: both
# agree on (0, -0.12, 0.005)). This is a fixed property of link2's own
# mesh and does NOT get mirrored -- mirroring only touches the joint
# origins that place link1/link2 relative to their parent, never what's
# inside link2's own body frame.
FOOT_SITE_POS = "0 -0.12 0.005"

# rectangle.STL's own bounding box, local Z axis only (mm) -- needed to
# reposition the mesh itself for mirrored legs, see mirror_origin().
RECTANGLE_Z_MIN_MM = 0.01910819
RECTANGLE_Z_MAX_MM = 57.540714

# ---------------------------------------------------------------
# body1.STL (chassis) -- measured directly from the exported mesh:
# bounding box X:[0,400] Y:[0,220] Z:[0,115] mm. Origin sits at one
# corner, same situation rectangle.STL was in.
#
# X and Z: hip_pos's X and Z values fall inside this range in a way
# that's physically plausible without any shift (e.g. legs mounted
# near the top of the block, one hip slightly behind the block's own
# X=0 edge) -- so no correction applied on those axes.
#
# Y: hip_pos is symmetric about Y=0 (left/right legs at roughly
# +-125mm, mirroring around the robot's centerline), but the mesh's
# own Y range is entirely positive (0 to 220mm) -- there's no
# corresponding negative side at all. That means the mesh's local
# origin is NOT on the centerline. Shifting it by -110mm (half the
# width) centers the block on the same Y=0 axis the legs mirror
# around.
#
# UNVERIFIED: this assumes body1's own part origin in Assem2.STEP is
# the same "body1 reference point" hip_pos is measured from for X and
# Z. Load quadruped_generated.xml in the viewer and check the block
# visually lines up with all 4 hip mounts before trusting this --
# if it's off, it'll most likely need an X and/or Z shift added the
# same way Y was handled here.
BODY1_Y_MIN_MM = 0.0
BODY1_Y_MAX_MM = 220.0
BODY1_MESH_POS = f"0 {-(BODY1_Y_MIN_MM + BODY1_Y_MAX_MM) / 2000.0} 0"

# Alumina, as assigned to body1 in SolidWorks -- solid ceramic alumina
# is ~3950 kg/m^3. Given as a density (not a fixed mass) so MuJoCo
# derives mass and inertia from the actual mesh volume automatically,
# same auto-computation the leg links already rely on implicitly.
# Revisit if the real chassis material/manufacturing ends up different
# from what's currently assigned in CAD -- solid alumina is unusual
# for a robot chassis and this may just be a CAD placeholder.
CHASSIS_DENSITY_KG_M3 = 3950


def mirror_origin(xyz_str):
    # Negate the LOCAL Z component -- under BASE_ROLL, local Z is the
    # axis that maps to world lateral (sideways). Reusing leg.urdf's
    # fixed joint origins as-is on the mirrored side pushes the leg
    # inward instead of outward (checked directly, see project history);
    # negating local Z here -- a plain sign flip on a fixed offset, not
    # a rotation -- fixes that without disturbing height or forward
    # direction.
    x, y, z = (float(v) for v in xyz_str.split())
    return f"{x} {y} {-z}"


def rectangle_mesh_pos(mirror):
    # rectangle.STL's own origin sits near one corner, not the mesh's
    # center -- its footprint occupies local Z in [Z_MIN, Z_MAX],
    # entirely on the positive side. That matches the unmirrored legs'
    # (positive) joint origin, but mirror_origin() above flips the
    # joint to negative for the mirrored legs, so the mesh must move
    # too or it ends up on the opposite side from where the leg
    # actually attaches. Fix: translate (not reflect) by -(Z_MIN+Z_MAX)
    # so the mesh's own range lands at [-Z_MAX,-Z_MIN] -- its mirror
    # image, reached by sliding the rigid mesh, never by flipping a
    # vertex, so shape and normals stay untouched.
    if not mirror:
        return "0 0 0"
    z_offset_m = -(RECTANGLE_Z_MIN_MM + RECTANGLE_Z_MAX_MM) / 1000.0
    return f"0 0 {z_offset_m}"


def parse_leg(path):
    root = ET.parse(path).getroot()

    def mesh_origin(link_name):
        origin = root.find(f".//link[@name='{link_name}']/visual/origin")
        return origin.get("xyz")

    def joint(name):
        j = root.find(f".//joint[@name='{name}']")
        return (
            j.find("origin").get("xyz"),
            j.find("limit").attrib,
        )

    return {
        "link1_mesh_origin": mesh_origin("link1"),
        "link2_mesh_origin": mesh_origin("link2"),
        "j1": joint("rectangle_to_link1"),
        "j2": joint("link1_to_link2"),
    }


def leg_body(name, cfg, geo):
    hx, hy, hz = cfg["hip_pos"]
    j1_origin, j1_limit = geo["j1"]
    j2_origin, j2_limit = geo["j2"]
    # gap-closing fix FIRST: pull the hinge in to sit exactly at the
    # rectangle mesh's own edge instead of 22mm past it (see project
    # history) -- must happen before mirroring, not after, or the
    # unconditional overwrite here would erase the mirror's sign flip
    x1, y1, _ = (float(v) for v in j1_origin.split())
    j1_origin = f"{x1} {y1} {RECTANGLE_Z_MAX_MM / 1000}"
    if cfg["mirror"]:
        j1_origin = mirror_origin(j1_origin)
        j2_origin = mirror_origin(j2_origin)
    rect_pos = rectangle_mesh_pos(cfg["mirror"])

    return f"""
    <body name="rectangle_{name}" pos="{hx} {hy} {hz}" euler="{BASE_ROLL} 0 0">
      <geom pos="{rect_pos}" type="mesh" contype="0" conaffinity="0" group="1" density="0" mesh="rectangle"/>
      <geom pos="{rect_pos}" type="mesh" mesh="rectangle"/>
      <body name="link1_{name}" pos="{j1_origin}">
        <joint name="rectangle_to_link1_{name}" range="{j1_limit['lower']} {j1_limit['upper']}" actuatorfrcrange="-100 100"/>
        <geom pos="{geo['link1_mesh_origin']}" type="mesh" contype="0" conaffinity="0" group="1" density="0" mesh="link1"/>
        <geom pos="{geo['link1_mesh_origin']}" type="mesh" mesh="link1"/>
        <body name="link2_{name}" pos="{j2_origin}">
          <site name="foot_site_{name}" type="sphere" size="0.02" pos="{FOOT_SITE_POS}" rgba="1 0 0 0.4"/>
          <joint name="link1_to_link2_{name}" range="{j2_limit['lower']} {j2_limit['upper']}" actuatorfrcrange="-100 100"/>
          <geom pos="{geo['link2_mesh_origin']}" type="mesh" contype="0" conaffinity="0" group="1" density="0" mesh="link2"/>
          <geom pos="{geo['link2_mesh_origin']}" type="mesh" mesh="link2"/>
        </body>
      </body>
    </body>
"""


def chassis_body(legs_xml):
    # pos="0 0 0": hip_pos already carries the height (z~0.09m) that
    # used to place each leg directly in world space, so starting the
    # chassis at the origin exactly reproduces the old world-fixed
    # geometry at t=0 -- nothing visually jumps when this lands, the
    # chassis just becomes free to move away from that pose afterward
    # (once something is actually driving/integrating it -- see the
    # note in simulate_quadruped.py about mj_forward vs mj_step).
    return f"""
    <body name="chassis" pos="0 0 0">
      <freejoint name="chassis_free"/>
      <geom name="chassis_visual" pos="{BODY1_MESH_POS}" type="mesh" contype="0" conaffinity="0" group="1" density="0" mesh="body1"/>
      <geom name="chassis_collision" pos="{BODY1_MESH_POS}" type="mesh" mesh="body1" density="{CHASSIS_DENSITY_KG_M3}"/>
{legs_xml}    </body>
"""


def build():
    geo = parse_leg(SOURCE_URDF)
    legs_xml = "".join(leg_body(name, cfg, geo) for name, cfg in LEGS.items())
    chassis_xml = chassis_body(legs_xml)
    sensors_xml = "".join(
        f'    <touch name="foot_force_{name}" site="foot_site_{name}"/>\n' for name in LEGS
    )

    # Floor: a standard horizontal plane (euler left at default, unlike
    # leg_generated.xml's 90-about-Y floor -- that rotation was needed
    # there because the single-leg test had no BASE_ROLL to already
    # orient "down" correctly; ours already does that at the mount, so
    # the floor itself needs no extra rotation). Height and footprint
    # come from the mechanism's own verified reachable workspace, same
    # numbers as the box-ground version this replaces.
    ground_z = -0.0638
    ground_xml = f'    <geom name="floor" type="plane" size="0.5 0.5 0.01" pos="0.15 0 {ground_z}" rgba="0.55 0.55 0.55 1"/>\n'

    xml = f"""<mujoco model="quadruped">
  <compiler angle="radian" meshdir="../meshes/"/>

  <asset>
    <mesh name="body1" content_type="model/stl" file="../meshes/body1.STL" scale="0.001 0.001 0.001"/>
    <mesh name="rectangle" content_type="model/stl" file="../meshes/rectangle.stl" scale="0.001 0.001 0.001"/>
    <mesh name="link1" content_type="model/stl" file="../meshes/link1.STL" scale="0.001 0.001 0.001"/>
    <mesh name="link2" content_type="model/stl" file="../meshes/link2.STL" scale="0.001 0.001 0.001"/>
  </asset>

  <worldbody>
{ground_xml}{chassis_xml}  </worldbody>

  <sensor>
{sensors_xml}  </sensor>
</mujoco>
"""

    with open(OUTPUT_XML, "w") as f:
        f.write(xml)

    print(f"Wrote {OUTPUT_XML} with chassis: body1.STL, legs: {', '.join(LEGS)}")
    print(f"Sites: {', '.join('foot_site_'+n for n in LEGS)}")
    print(f"Touch sensors: {', '.join('foot_force_'+n for n in LEGS)}")


if __name__ == "__main__":
    build()