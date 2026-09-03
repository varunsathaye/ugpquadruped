"""
leg_config.py

Single source of truth for the 4-leg layout, imported by both
build_quadruped.py and simulate_quadruped.py so they can't drift out
of sync.

hip_pos values below come from Assem2.STEP (the "modelassemble1"
leg-subassembly placements relative to a body1 reference point).
FR/FL/BR/BL labels are a guess (see build_quadruped.py) -- verify
against your physical robot and rename the dict keys if they're
wrong; nothing else here depends on the labels being correct.

"mirror" is kept as a label only (which side a leg is physically on)
-- it no longer changes the rotation or the IK; see the note below
hip_rpy for why.
"""

import math

LEGS = {
    "FR": {"hip_pos": (0.17956, -0.12527, 0.09025), "mirror": False},
    "BR": {"hip_pos": (-0.02044, -0.12486, 0.09025), "mirror": False},
    "FL": {"hip_pos": (0.20720, 0.12445, 0.09025), "mirror": True},
    "BL": {"hip_pos": (0.00720, 0.12486, 0.09025), "mirror": True},
}

# ---------------------------------------------------------------
# Why there's no per-leg mirror rotation here (there used to be):
#
# The fixed hinge offset baked into leg.urdf's rectangle_to_link1
# joint (0.03, 0.03, 0.08) has a component along the same local axis
# that any extra 180 mirror rotation flips the sign of -- so on top
# of correctly mirroring the gait direction, it was ALSO flipping
# that fixed 0.03m offset, pushing the mirrored legs 0.06m lower
# than the others (checked directly: link1_FR/BR sat at z=0.1202,
# link1_FL/BL at z=0.0602 -- a clean 0.06 = 2x0.03 gap).
#
# Since a rotation can't mirror this leg without some cost (proven
# earlier: no proper rotation preserves both forward and vertical
# while flipping lateral), and a height mismatch is a worse cost
# than a cosmetic one, all 4 legs now get the SAME rotation
# (BASE_ROLL only). Every leg's forward and vertical response is
# now identical -- verified: all 4 sit at the same z at rest, and
# simulate_quadruped.py no longer needs to negate anything. The
# tradeoff is that the left-side knees bend the same absolute
# direction as the right-side ones rather than mirroring outward --
# purely cosmetic, revisit once there's a real chassis mesh to
# check the visual against.
# ---------------------------------------------------------------
BASE_ROLL = math.pi / 2


def hip_rpy(mirror=None):
    return (BASE_ROLL, 0.0, 0.0)
