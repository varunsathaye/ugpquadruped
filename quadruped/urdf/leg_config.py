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

# ---------------------------------------------------------------
# TUNING LEG SPACING -- this dict is the only place to change it.
# Because every leg is mounted with BASE_ROLL (pi/2 about X), the
# three hip_pos components map to world axes as:
#
#   x -> fore/aft.  WHEELBASE = front x - rear x. The rectangle
#        bracket extends +0.095m FORWARD from x, so x is the
#        bracket's REAR edge.
#   y -> left/right. TRACK = |y| * 2. The bracket extends OUTWARD
#        from y (away from the centerline), so |y| is exactly the
#        bracket's inner face -- set it to the chassis half-width
#        to sit flush. Sign must agree with "mirror": negative y
#        for mirror=False (right side), positive for mirror=True.
#   z -> height. The bracket extends +0.097m UPWARD from z, so z
#        is the bracket's BOTTOM edge.
#
# Values below are a symmetric layout fitted to body1.STL, whose
# measured extents are X[0, 0.400] Y[-0.110, 0.110] Z[0, 0.115]:
#   x = 0.305 / 0.000 puts each bracket flush with the chassis's
#       front and rear faces (wheelbase 0.305m, the most the
#       0.400m body allows with a 0.095m bracket).
#   y = +-0.110 is the chassis side face, closing the ~15mm air
#       gap the old Assem2-derived +-0.125 values left.
# The old values were read straight out of Assem2.STEP and were
# neither symmetric (left pair sat 27.6mm forward of the right)
# nor consistent with body1.STL's own extents.
# ---------------------------------------------------------------
LEGS = {
    "FR": {"hip_pos": (0.30500, -0.11000, 0.09025), "mirror": False},
    "BR": {"hip_pos": (0.00000, -0.11000, 0.09025), "mirror": False},
    "FL": {"hip_pos": (0.30500, 0.11000, 0.09025), "mirror": True},
    "BL": {"hip_pos": (0.00000, 0.11000, 0.09025), "mirror": True},
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
