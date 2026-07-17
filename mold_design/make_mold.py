# ============================================================================
# Parametric two-plate compression mold — CONCENTRIC optical silicone dome
# for the SparkFun Qwiic Mini ToF Imager (VL53L5CX, SEN-19013), Sorta-Clear 40.
#
# ORIGIN = the sensor's OPTICAL CENTRE (Rx aperture), located from ground
# truth, at Z = sensor top surface (+1.5 mm above the board):
#   * ST DS13754 Fig.21: Rx aperture = "OPTICAL CENTRE", 2.0 mm off the
#     package mechanical centre along its length, 0.1 mm across; Tx-Rx = 4.0
#   * SparkFun Eagle .brd: aperture circle at package (-2.0, -0.1); element
#     U1 at (12.7, 8.89) rot MR0  ->  Rx at BOARD (14.70, 8.79)
#   * board 25.4 x 12.7; standoff holes (2.54,2.54) & (22.86,2.54) dia 3.30
#
# Both dome surfaces are spheres about the origin -> zero-power dome port:
# no FoV remap, uniform 3 mm wall, single global range bias, Fresnel
# back-reflection retro-directed to Tx/Rx sources rather than across them.
#
# MOUNTING CHANGE (forced by geometry): the +X board standoff hole is only
# 10.3 mm from the Rx centre; a concentric dome large enough to clear the
# package would be pierced by it. The lens therefore fastens with 4 corner
# screws into the mount, and gets shallow head-relief pockets over the two
# board screws instead of through-holes.
#
# Run:  /Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd make_mold.py
# Z=0 = parting plane = the part's flat board-contact face; dome-DOWN.
# ============================================================================

import math
import os
import FreeCAD as App
import Part
import MeshPart
from FreeCAD import Vector

OUT = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# Board interface, expressed relative to the OPTICAL CENTRE (all mm)
# ----------------------------------------------------------------------------
rx_board = (14.70, 8.79)                 # Rx optical centre in board coords
board_x0, board_x1 = -14.70, 10.70       # board edges rel. optical centre
board_y0, board_y1 = -8.79, 3.91
board_holes = [(-12.16, -6.25), (8.16, -6.25)]   # existing standoffs
# Integral silicone snap-plugs cast into the flange underside: they press
# through the board's own two 3.30 mm holes and mushroom-lock beneath it.
# !!! MEASURE YOUR PCB THICKNESS with calipers and set pcb_t before printing:
# the barb must sit just below the board's bottom face to snap and retain.
pcb_t        = 1.6      # PCB thickness (SparkFun std 1.6; some minis are 0.8)
plug_stem_d  = 3.4      # 0.1 interference vs the 3.30 hole
plug_barb_d  = 4.6      # mushroom bulb (stretch ratio 1.4x through the hole)
plug_barb_h  = 1.0
plug_tip_h   = 1.0      # taper to a Ø2 tip for easy insertion
plug_vent_d  = 1.0      # air escape so the plug cavity fills bubble-free
pkg_cx, pkg_cy = -2.0, 0.1               # package mech centre rel. Rx
pkg_hx, pkg_hy = 3.2, 1.5                # package half-extents (6.4 x 3.0)
sensor_h       = 1.5
tx_pos         = (-4.0, 0.2)             # Tx aperture rel. Rx
fov_half_diag  = 31.7                    # deg (45x45 FoV diagonal)
tx_excl_half   = 31.0                    # deg, Tx exclusion cone half-angle

# ----------------------------------------------------------------------------
# The cast lens (all spheres centered at origin, z = -sensor_h)
# ----------------------------------------------------------------------------
flange_t     = 3.0
flange_rad   = 2.0
# Flange is SYMMETRIC about the dome/optical axis: -X edge = board edge, and
# +X extends 4.0 past the board edge (mount face must be flush all around).
fl_x0, fl_x1 = -14.70, 14.70
fl_y0, fl_y1 = -12.00, 12.00
dome_base_dia = 19.0           # dome dia at flange top
wall          = 3.0            # uniform radial optical wall
cav_r         = 6.0            # cavity skirt cylinder radius
cav_cx, cav_cy = 0.0, 0.0      # skirt ON the optical axis: rotationally
                               # symmetric boss, uniform junction at z=-4.29
                               # (possible because cav_R grew to 6.62)
screw_hole_dia = 2.7           # M2.5 corner fasteners into the mount
# y=+-10 (not +-8.1): keeps a >=0.5 mm web to the snap-plug cavities
screw_pos = [(sx * 12.2, sy * 10.0) for sx in (-1, 1) for sy in (-1, 1)]

# Derived dome geometry: sphere centre must sit at z = -flange_t + ... such
# that centre height = sensor top: R - h = flange_t - sensor_h
dome_a = dome_base_dia / 2.0
_c     = flange_t - sensor_h                       # 1.5
dome_h = (-(2*_c - 0) + math.sqrt((2*_c)**2 + 4*dome_a**2)) / 2.0 \
         if False else (-3 + math.sqrt(9 + 4*dome_a**2)) / 2.0   # h^2+2c h-a^2=0, c=1.5
dome_R  = (dome_a**2 + dome_h**2) / (2.0 * dome_h)
cav_R   = dome_R - wall
opt_z   = -sensor_h                                # common sphere centre z

# ----------------------------------------------------------------------------
# The mold
# ----------------------------------------------------------------------------
plate_w   = 72.0
plate_c   = (0.0, 0.0)         # plates centred on the dome / optical axis
bot_t     = 16.0
top_t     = 12.0
land_w    = 3.0
gutter_w  = 3.0
gutter_d  = 2.5
vent_w    = 2.0
vent_d    = 0.5
riser_dia = 5.0
riser_pos = [(-10.5, 4.0), (10.5, 4.0)]      # mirrored pair
funnel_dia, funnel_depth = 14.0, 5.0
telltale_dia = 2.0
telltale_pos = [(-19.2, 8.0), (19.2, 8.0)]   # mirrored pair, over the gutter
bolt_dia, bolt_off = 5.5, 29.0
nut_af, nut_pocket_d = 8.0, 4.5
dowel_press, dowel_slip, dowel_depth = 3.90, 4.20, 8.0
# 4 dowels at the edge midpoints — visually even. ONE is offset 1.5 mm
# (industry-standard offset leader pin): the plates still cannot be
# assembled 90/180 deg rotated, which would misplace the snap-plug cavities.
dowel_pos = [(0.0, 26.0), (26.0, 0.0), (0.0, -26.0), (-26.0, 1.5)]

half = plate_w / 2.0
doc = App.newDocument("mold")


def rounded_rect(x0, y0, x1, y1, r, z, h):
    a = Part.makeBox(x1 - x0 - 2 * r, y1 - y0, h, Vector(x0 + r, y0, z))
    b = Part.makeBox(x1 - x0, y1 - y0 - 2 * r, h, Vector(x0, y0 + r, z))
    s = a.fuse(b)
    for cx in (x0 + r, x1 - r):
        for cy in (y0 + r, y1 - r):
            s = s.fuse(Part.makeCylinder(r, h, Vector(cx, cy, z)))
    return s


def hex_prism(af, height, center):
    r = af / math.sqrt(3.0)
    pts = [Vector(center.x + r * math.cos(math.radians(60 * i + 30)),
                  center.y + r * math.sin(math.radians(60 * i + 30)),
                  center.z) for i in range(7)]
    return Part.Face(Part.makePolygon(pts)).extrude(Vector(0, 0, height))


def poka_yoke_corner(solid, z0, z1):
    box = Part.makeBox(8, 8, z1 - z0, Vector(-4, -4, z0))
    box.rotate(Vector(0, 0, 0), Vector(0, 0, 1), 45)
    box.translate(Vector(plate_c[0] + half, plate_c[1] + half, 0))
    return solid.cut(box)


# ============================================================================
# BOTTOM (CAVITY) PLATE — z in [-bot_t, 0]
# ============================================================================
bot = Part.makeBox(plate_w, plate_w, bot_t,
                   Vector(plate_c[0] - half, plate_c[1] - half, -bot_t))

bot = bot.cut(rounded_rect(fl_x0, fl_y0, fl_x1, fl_y1, flange_rad,
                           -flange_t, flange_t))

# Dome cavity: sphere about the optical centre
sph_cz = -flange_t - dome_h + dome_R           # == opt_z by construction
bot = bot.cut(Part.makeSphere(dome_R, Vector(0, 0, sph_cz)))

# Rounded-rect overflow gutter ring around the land
g_out = rounded_rect(fl_x0 - land_w - gutter_w, fl_y0 - land_w - gutter_w,
                     fl_x1 + land_w + gutter_w, fl_y1 + land_w + gutter_w,
                     flange_rad + land_w + gutter_w, -gutter_d, gutter_d)
g_in = rounded_rect(fl_x0 - land_w, fl_y0 - land_w,
                    fl_x1 + land_w, fl_y1 + land_w,
                    flange_rad + land_w, -gutter_d, gutter_d)
bot = bot.cut(g_out.cut(g_in))

# Vent slots at the four flange-edge midpoints
vents = [
    Part.makeBox(land_w + 2, vent_w, vent_d, Vector(fl_x1 - 1, -vent_w/2, -vent_d)),
    Part.makeBox(land_w + 2, vent_w, vent_d, Vector(fl_x0 - land_w - 1, -vent_w/2, -vent_d)),
    Part.makeBox(vent_w, land_w + 2, vent_d, Vector(plate_c[0] - vent_w/2, fl_y1 - 1, -vent_d)),
    Part.makeBox(vent_w, land_w + 2, vent_d, Vector(plate_c[0] - vent_w/2, fl_y0 - land_w - 1, -vent_d)),
]
for v in vents:
    bot = bot.cut(v)

# Corner screw-hole core pins (kiss the parting plane)
for (px, py) in screw_pos:
    bot = bot.fuse(Part.makeCylinder(screw_hole_dia / 2.0, flange_t,
                                     Vector(px, py, -flange_t)))

# Bolts + hex nut pockets
for sx in (-1, 1):
    for sy in (-1, 1):
        c = Vector(plate_c[0] + sx * bolt_off, plate_c[1] + sy * bolt_off, 0)
        bot = bot.cut(Part.makeCylinder(bolt_dia / 2.0, bot_t,
                                        Vector(c.x, c.y, -bot_t)))
        bot = bot.cut(hex_prism(nut_af, nut_pocket_d, Vector(c.x, c.y, -bot_t)))

# Dowels
for (dx, dy) in dowel_pos:
    bot = bot.cut(Part.makeCylinder(dowel_press / 2.0, dowel_depth,
                                    Vector(dx, dy, -dowel_depth)))

# Pry notches
for sx in (-1, 1):
    ex = plate_c[0] + sx * half
    bot = bot.cut(Part.makeBox(6, 14, 3, Vector(ex - (0 if sx < 0 else 6),
                                                plate_c[1] - 7, -3)))

bot = poka_yoke_corner(bot, -bot_t, 0)

# ============================================================================
# TOP (CORE) PLATE — z in [0, top_t]
# ============================================================================
top = Part.makeBox(plate_w, plate_w, top_t,
                   Vector(plate_c[0] - half, plate_c[1] - half, 0))

# Core boss: concentric spherical ceiling + cylindrical skirt
ball = Part.makeSphere(cav_R, Vector(0, 0, opt_z))
skirt = Part.makeCylinder(cav_r, 10.0, Vector(cav_cx, cav_cy, -10.0))
top = top.fuse(ball.common(skirt))

# Snap-plug cavities over the board's two holes: stem, mushroom barb,
# insertion taper, and a small vent so displaced air escapes upward.
for (px, py) in board_holes:
    z = 0.0
    top = top.cut(Part.makeCylinder(plug_stem_d / 2.0, pcb_t,
                                    Vector(px, py, z)))
    z += pcb_t
    top = top.cut(Part.makeCylinder(plug_barb_d / 2.0, plug_barb_h,
                                    Vector(px, py, z)))
    z += plug_barb_h
    top = top.cut(Part.makeCone(plug_barb_d / 2.0, 1.0, plug_tip_h,
                                Vector(px, py, z)))
    z += plug_tip_h
    top = top.cut(Part.makeCylinder(plug_vent_d / 2.0, top_t - z,
                                    Vector(px, py, z)))

# Risers with funnels
for (x, y) in riser_pos:
    top = top.cut(Part.makeCylinder(riser_dia / 2.0, top_t, Vector(x, y, 0)))
    top = top.cut(Part.makeCone(riser_dia / 2.0, funnel_dia / 2.0, funnel_depth,
                                Vector(x, y, top_t - funnel_depth)))

# Telltale vents over the gutter
for (x, y) in telltale_pos:
    top = top.cut(Part.makeCylinder(telltale_dia / 2.0, top_t, Vector(x, y, 0)))

# Bolts
for sx in (-1, 1):
    for sy in (-1, 1):
        top = top.cut(Part.makeCylinder(bolt_dia / 2.0, top_t,
                                        Vector(plate_c[0] + sx * bolt_off,
                                               plate_c[1] + sy * bolt_off, 0)))

# Dowel slip holes
for (dx, dy) in dowel_pos:
    top = top.cut(Part.makeCylinder(dowel_slip / 2.0, top_t, Vector(dx, dy, 0)))

top = poka_yoke_corner(top, 0, top_t)

# ============================================================================
# Design-rule checks (all geometry is checked, not assumed)
# ============================================================================
assert bot.isValid() and top.isValid(), "Boolean result invalid"

print("Dome: base D%.1f h=%.3f R=%.4f | cavity R=%.4f | wall=%.2f (uniform)"
      % (dome_base_dia, dome_h, dome_R, cav_R, dome_R - cav_R))
print("Sphere centre z=%.3f (must equal -sensor_h=%.3f)" % (sph_cz, opt_z))

# 1. cavity skirt fully inside the cavity sphere
far = math.hypot(cav_cx, cav_cy) + cav_r
print("Skirt far point %.2f vs cavity sphere R %.2f -> %s (margin %.2f)"
      % (far, cav_R, "OK" if far < cav_R else "FAIL", cav_R - far))

# 2. package clearance inside the skirt
worst = 0.0
for sx in (-1, 1):
    for sy in (-1, 1):
        d = math.hypot(pkg_cx + sx * pkg_hx - cav_cx,
                       pkg_cy + sy * pkg_hy - cav_cy)
        worst = max(worst, d)
print("Package corner max %.2f vs skirt r %.2f -> %s (clearance %.2f)"
      % (worst, cav_r, "OK" if worst < cav_r else "FAIL", cav_r - worst))

# 3. Rx FoV: cap angular radius vs FoV diagonal
rim_lat = min(cav_r + math.hypot(cav_cx, cav_cy),
              cav_r - 0 + 0) + 0  # conservative: nearest rim lateral
rim_lat = cav_r - math.hypot(cav_cx, cav_cy)  # worst (nearest) rim offset
cap_ang = math.degrees(math.asin(min(1.0, (rim_lat + 2*math.hypot(cav_cx, cav_cy)) / cav_R))) \
    if False else math.degrees(math.asin(min(1.0, rim_lat / cav_R)))
print("Cap min angular radius %.1f deg vs FoV diag %.1f deg -> %s"
      % (math.degrees(math.asin(min(1.0, rim_lat / cav_R))), fov_half_diag,
         "OK" if math.asin(min(1.0, rim_lat / cav_R)) > math.radians(fov_half_diag) else "CHECK"))

# 4. Tx exclusion cone lands on the spherical cap (worst outboard ray)
txx, txy = tx_pos
dirx = -math.sin(math.radians(tx_excl_half))
dirz = -math.cos(math.radians(tx_excl_half))
# ray-sphere intersection in the XZ plane through Tx
b = 2 * (txx * dirx + (0 - 0) )  # sphere centred at origin laterally
b = 2 * (txx * dirx)
cq = txx * txx - cav_R * cav_R
t = (-b + math.sqrt(b * b - 4 * cq)) / 2.0
hit_x, hit_z = txx + t * dirx, opt_z + t * dirz
d_hit = math.hypot(hit_x - cav_cx, txy - cav_cy)
print("Tx worst ray hits cap at lateral %.2f (skirt r %.2f) -> %s"
      % (d_hit, cav_r, "OK" if d_hit < cav_r else "FAIL"))

# 5. corner screw pins clear the dome base circle AND the snap-plug cavities
for (px, py) in screw_pos:
    cl = math.hypot(px, py) - screw_hole_dia / 2.0 - dome_a
    pg = min(math.hypot(px - qx, py - qy) for (qx, qy) in board_holes) \
        - screw_hole_dia / 2.0 - plug_stem_d / 2.0
    print("Screw pin (%6.2f,%6.2f) dome clr %.2f %s | plug web %.2f %s"
          % (px, py, cl, "OK" if cl > 0.6 else "FAIL",
             pg, "OK" if pg > 0.5 else "FAIL"))

# 6. dowels must block 90/180/270-deg mis-assembly
for ang in (90, 180, 270):
    a = math.radians(ang)
    fits = all(any(math.hypot(dx*math.cos(a)-dy*math.sin(a) - ex,
                              dx*math.sin(a)+dy*math.cos(a) - ey) < 0.3
                   for (ex, ey) in dowel_pos) for (dx, dy) in dowel_pos)
    print("Rotated assembly %3d deg: %s" % (ang, "FAIL (fits!)" if fits else "BLOCKED OK"))

# 7. full clearance matrix: every parting-plane feature vs every other
features = (
    [("pin", p, screw_hole_dia/2) for p in screw_pos]
    + [("plug", p, plug_stem_d/2) for p in board_holes]
    + [("riser", p, riser_dia/2) for p in riser_pos]
    + [("telltale", p, telltale_dia/2) for p in telltale_pos]
    + [("dowel", p, dowel_slip/2) for p in dowel_pos]
    + [("bolt", (plate_c[0]+sx*bolt_off, plate_c[1]+sy*bolt_off), bolt_dia/2)
       for sx in (-1, 1) for sy in (-1, 1)])
worst_pair, worst_gap = None, 1e9
for i in range(len(features)):
    for j in range(i+1, len(features)):
        (n1, p1, r1), (n2, p2, r2) = features[i], features[j]
        gap = math.hypot(p1[0]-p2[0], p1[1]-p2[1]) - r1 - r2
        if gap < worst_gap:
            worst_gap, worst_pair = gap, (n1, n2)
        assert gap > 0.5, "CLEARANCE FAIL: %s%s vs %s%s gap %.2f" % (n1, p1, n2, p2, gap)
print("Clearance matrix: %d feature pairs, all > 0.5 mm (worst %.2f: %s-%s) OK"
      % (len(features)*(len(features)-1)//2, worst_gap, *worst_pair))

cap_vol = math.pi * dome_h**2 * (dome_R - dome_h / 3.0)
fl_area = (fl_x1 - fl_x0) * (fl_y1 - fl_y0) - (4 - math.pi) * flange_rad**2
part_vol = fl_area * flange_t + cap_vol \
    - math.pi * cav_r**2 * 3.0 - 2.0/3.0 * math.pi * cav_R**3 * 0.5  # rough
print("Silicone shot ~= %.1f mL (mix >= 20 g)" % (part_vol / 1000.0))

# ============================================================================
# Export
# ============================================================================
o_bot = doc.addObject("Part::Feature", "BottomCavityPlate"); o_bot.Shape = bot
o_top = doc.addObject("Part::Feature", "TopCorePlate");      o_top.Shape = top
doc.recompute()

Part.export([o_bot], os.path.join(OUT, "bottom_cavity_plate.step"))
Part.export([o_top], os.path.join(OUT, "top_core_plate.step"))
Part.export([o_bot, o_top], os.path.join(OUT, "mold_assembly.step"))
for shape, name in ((bot, "bottom_cavity_plate"), (top, "top_core_plate")):
    MeshPart.meshFromShape(Shape=shape, LinearDeflection=0.03,
                           AngularDeflection=0.35).write(
        os.path.join(OUT, name + ".stl"))
print("Exported STEP + STL to", OUT)
