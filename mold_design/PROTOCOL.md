# Dome Lens Casting Protocol — Simple Version

## Equipment
1. The two printed mold plates (bottom = dome cavity, top = boss + funnels)
2. 4× M5 bolts + nuts, 2× wing nuts optional
3. 4× Ø4 steel dowel pins (press into the bottom plate once, they stay)
4. Sorta-Clear 40 part A and B, microscale, mixing cup (wide, tall), stick
5. Vacuum chamber, pressure pot (55–60 psi), compressor
6. Ease Release 200 (or other platinum-safe release)
7. Nitrile gloves, 99% IPA, non-fibrous wipes, fresh razor blade

## One-time mold prep (first use only)
1. **Measure your PCB thickness with calipers.** If it is not 1.6 mm, set
   `pcb_t` in make_mold.py, regenerate, and reprint the TOP plate.
2. Seal both plates (FDM: brush thin epoxy/XTC-3D; SLA: full UV post-cure
   + 2 h at 60 °C).
3. Polish the dome cavity (bottom plate) and the boss cap (top plate) to
   2000+ grit, then buff. These two surfaces ARE your optics.
4. Press the 4 dowel pins into the bottom plate holes (one is offset 1.5 mm
   on purpose — the top plate only fits one way; never force it).
5. Cast one small test patch of silicone against a printed scrap to confirm
   it cures tack-free (platinum silicone + resin can inhibit).

## Every cast
1. Wipe plates with IPA, let dry. Mist release on all cavity surfaces, buff
   lightly. Gloves on.
2. Weigh **20 g A : 2 g B** (10:1). Mix thoroughly ~3 min without whipping.
3. Degas the cup at 28–29 inHg until the foam rises, collapses, and the
   surface goes quiet (~4 min). Vent slowly.
4. Bottom plate flat and level. Pour a thin, low stream into the dome
   cavity until the flange recess is full and slightly proud of the land.
5. Degas the filled open bottom plate 2–3 min. Top up if the level drops.
6. Lower the top plate at a slight tilt, dowels engaged, letting the boss
   enter the pool edge-first. Press down slowly and evenly.
   **STOP CHECK:** silicone must rise in BOTH funnels and appear at BOTH
   tiny telltale holes. If not, open, top up, repeat.
7. Bolt the 4 corners in a cross pattern until the plates seat metal-to-metal.
8. Top up both funnels with leftover silicone.
9. Straight into the pressure pot: **55–60 psi, 16 h, level, funnels up.**
   Never vacuum the closed mold.
10. Vent the pot slowly. Unbolt. Pry gently at the two side notches. Flex
    the part out of the bottom plate; peel the snap-plugs out of the top
    plate (they stretch — pull straight).
11. Trim with a fresh blade: two riser stubs, gutter flash ring, and the
    two thin vent whiskers on the snap-plugs. Done.

## Install on the sensor
1. Orient the lens: the two snap-plugs match the board's two holes — it
   only fits one way.
2. Press the plugs through the PCB holes until the barbs click under.
3. (Optional, rigid setup) Screw the 4 corner M2.5 holes into your mount.
   Mount face must be flush with the board's top surface.

## If something goes wrong
- Bubbles at the dome base → you vacuumed the closed mold, or closed it too
  fast. Follow steps 5–6 exactly.
- Tacky surface → cure inhibition: re-do mold prep steps 2 and 5.
- Cloudy dome → polish the mold optics (prep step 3); the silicone copies
  every scratch.
- Plug won't click → wrong `pcb_t`; re-measure the board, reprint top plate.
