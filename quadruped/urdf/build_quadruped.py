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


def build():
    geo = parse_leg(SOURCE_URDF)
    legs_xml = "".join(leg_body(name, cfg, geo) for name, cfg in LEGS.items())
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
    <mesh name="rectangle" content_type="model/stl" file="../meshes/rectangle.stl" scale="0.001 0.001 0.001"/>
    <mesh name="link1" content_type="model/stl" file="../meshes/link1.STL" scale="0.001 0.001 0.001"/>
    <mesh name="link2" content_type="model/stl" file="../meshes/link2.STL" scale="0.001 0.001 0.001"/>
  </asset>

  <worldbody>
{ground_xml}{legs_xml}  </worldbody>

  <sensor>
{sensors_xml}  </sensor>
</mujoco>
"""

    with open(OUTPUT_XML, "w") as f:
        f.write(xml)

    print(f"Wrote {OUTPUT_XML} with legs: {', '.join(LEGS)}")
    print(f"Sites: {', '.join('foot_site_'+n for n in LEGS)}")
    print(f"Touch sensors: {', '.join('foot_force_'+n for n in LEGS)}")


if __name__ == "__main__":
    build()
