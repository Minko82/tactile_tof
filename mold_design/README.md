# Concentric Optical Dome Mold — SparkFun Qwiic Mini ToF Imager (VL53L5CX)

Two-plate compression mold for a **zero-power concentric dome port** in
Sorta-Clear 40 for the SparkFun Qwiic Mini ToF Imager (SEN-19013). Both dome
surfaces are spheres centred on the sensor's **optical centre (Rx aperture)**:
rays from the Rx cross both interfaces at normal incidence, so the dome adds
no FoV remap, a single uniform range bias (2·(n_g−1)·3 mm ≈ 2.5 mm,
calibrated once), and retro-directed rather than cross-coupled Fresnel
reflections.

## Optical-centre ground truth (why the dome is off the package centre)

- ST DS13754 Fig. 21: the **Rx aperture is the "OPTICAL CENTRE"**, 2.0 mm off
  the package mechanical centre along its length (Tx–Rx = 4.0 mm), 0.1 mm
  across.
- SparkFun's Eagle package draws that aperture at package (−2.0, −0.1); with
  U1 at (12.7, 8.89) mirrored, the **Rx lands at board (14.70, 8.79)**.
- All part/mold coordinates below use the Rx as origin, Z = sensor top.

Key derived geometry (all verified by design-rule checks in the script):
dome base Ø19 × 8.12 high, outer sphere **R 9.618**, inner sphere **R 6.618**,
both centred at the Rx, **uniform 3.00 mm wall**; cavity skirt Ø12.0 cylinder
**on the optical axis** (rotationally symmetric boss, uniform rim at
z = −4.29) clears the full package by 0.56 mm and stays 0.62 mm inside the
inner sphere; the spherical cap covers the 45°×45° FoV diagonal (65°
available vs 31.7° needed) **and** the Tx exclusion cone (worst outboard Tx
ray lands on the cap at lateral 5.86 vs 6.0 available).

## Mounting change (forced by the 2 mm Rx offset)

The board's +X standoff hole sits only 10.3 mm from the Rx — a concentric
dome large enough to clear the package would be pierced by it. Therefore:

- The lens fastens with **four M2.5 corner screws into the mount**
  (holes at (±12.2, ±10) rel. Rx — a fully symmetric pattern about the dome axis, positioned to keep a 0.7 mm web to the snap-plugs), *not* through the board.
- The lens stabilizes **directly on the board** with two integral silicone
  snap-plugs cast into the flange underside at the board's own hole
  positions, (−12.16, −6.25) and (8.16, −6.25): Ø3.4 stem (0.1 mm
  interference in the Ø3.30 holes), Ø4.6 mushroom barb that stretches
  through and locks beneath the PCB, Ø2 insertion tip. **Measure your PCB
  thickness and set `pcb_t` (default 1.6) before printing** — the barb must
  clear the board's bottom face. Note: the board has exactly two holes (per
  the Eagle file), so board-level mounting is two-point; the four corner
  screws provide the rigid four-point clamp into the mount.
- Flange 29.4 × 24.0 × 3, symmetric about the dome axis, overhangs the board edges (+X 4.0, −Y 3.2, +Y 8.1 mm)
  — the **mount face must be flush with the board's top surface** around the
  perimeter (board recessed in a pocket, as in the UR5 mount).
- The snap-plugs sit asymmetrically about the dome axis (the board's holes
  are symmetric about the *package*, which is 2 mm off the optical centre) —
  this doubles as poka-yoke: the lens only fits the board one way.

## Files

| File | What it is |
|---|---|
| `bottom_cavity_plate.step` / `.stl` | Cavity plate (dome + flange + gutter) |
| `top_core_plate.step` / `.stl` | Core plate (boss, risers, telltales) |
| `mold_assembly.step` | Both plates in assembled casting position |
| [make_mold.py](make_mold.py) | Parametric generator — edit params, rerun |
| [PROTOCOL.md](PROTOCOL.md) | Simple step-by-step casting protocol |
| `previews/` | Rendered views, cross-section, feature alignment map |

Regenerate after editing parameters:

```sh
/Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd make_mold.py
```

## Mold mechanics (why each feature exists)

- **Dome-down, open-face architecture** — silicone is degassed in the mold
  *with a free surface* (the only configuration where vacuum works), then the
  core plate squeezes closed. Never vacuum a closed filled mold: trapped gas
  expands ~7×, expels silicone through the vents, and 35,000 cps silicone
  cannot flow back — that was the original dome-base void failure.
- **Sealing land (3 mm) + 4 vent slots (2 × 0.5 mm)** at the flange edge
  midpoints — restricted squeeze-out paths into a **rounded-rect overflow
  gutter (3 wide × 2.5 deep)** that follows the flange outline.
- **Two Ø5 risers with Ø14 funnels** at (±10.5, 4) — pour relief,
  syringe port, and feed reservoirs during the pressure-pot cure.
- **Two Ø2 telltale vents** over the gutter — silicone appearing there is
  your visual full-squeeze-out confirmation before bolting.
- **Core boss = spherical cap (R 6.618) + coaxial cylindrical skirt (Ø12.0)** — the
  cap is the optical surface (conveniently convex, which polishes far more
  easily than a concave pocket); the skirt only clears the package.
- **Core pins Ø2.7 ×4** mold the corner screw holes; **snap-plug cavities
  ×2** in the top plate (stem + barb + taper + Ø1 vent — the vent lets air
  escape upward as silicone fills the plug, then trims off as a whisker).
- **4× Ø4 dowels at the edge midpoints** — visually even pattern; one pin
  is offset 1.5 mm (offset-leader-pin standard) so the plates physically
  cannot assemble rotated. Press-fit in bottom, slip-fit through top.
- 4× **M5 through-bolts** with hex-nut pockets underneath; **pry notches** at
  the parting line; **chamfered corner** on both plates as an orientation key.
- Plates 72 × 72; bottom 16 thick, top 12 thick.

## Fabrication notes (critical for optics)

1. **Print orientation:** bottom plate cavity-face UP, top plate boss UP —
   optical tool faces must never touch supports.
2. **Surface finish:** the silicone replicates the mold at micron scale. Sand
   the dome cavity and boss end-face to 2000+ grit and polish, or self-level
   with a thin clear coat (UV resin re-cure / XTC-3D). FDM molds must be
   sealed regardless to kill porosity.
3. **SLA cure inhibition:** Sorta-Clear 40 is platinum-cure. Post-cure SLA
   molds hard (full UV + ≥2 h @ 60 °C), apply a barrier/release coat, and
   run a silicone test patch before a real shot.
4. **Release:** platinum-safe (Ease Release 200), light mist, buffed.

## Casting protocol

1. Mix Sorta-Clear 40 (10A:1B by weight, ≥20 g — shot is ~2.6 mL plus
   gutter/risers); stir well, don't whip.
2. Vacuum degas the cup at 28–29 inHg (wide cup, ≥4× headspace) until the
   foam rises, collapses, and the surface goes quiet.
3. Mold open and level, release applied. Pour slowly into the dome cavity in
   a thin low stream until the flange recess is slightly proud.
4. Degas the **open** filled bottom plate 2–3 min. Top up if the level drops.
5. Lower the top plate slowly at a slight tilt (dowels engaged) so the boss
   enters the pool edge-first and sweeps air sideways. Press evenly until
   silicone rises in both riser funnels and shows at the telltale vents.
6. Bolt cross-pattern until the land seats. Top up both funnels.
7. **Cure in a pressure pot at 55–60 psi for the full 16 h** — residual
   microbubbles dissolve (Henry's law) and the risers feed the shrinkage.
   No vacuum ever touches the closed mold.
8. Vent slowly, unbolt, pry at the notches, flex the part out. Trim riser
   stubs and gutter flash with a fresh blade.
