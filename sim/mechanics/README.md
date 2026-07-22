# Custom fingertip mechanics

This workflow is independent of the ToF simulator. It validates a positive
silicone body, creates a tetrahedral mesh, maps the original coating surface
into that mesh, runs a deterministic Newton VBD touch, and exports mechanical
ground truth.

## Prepare an asset

Install the small preparation extras in the Python environment that runs
Newton (`sim/mechanics-requirements.txt`), then run:

```powershell
python sim/scripts/prepare_fingertip.py `
  --stl "sim/assets/new_fingertip_custom/custom.stl" `
  --units mm `
  --target-edge-mm 1.0 `
  --output-dir "sim/assets/new_fingertip_custom/prepared" `
  --regions-config "sim/config/mechanics/regions/custom_fingertip_regions.json"
```

The command creates `surface.stl`, `volume.msh`, `asset.json`, `regions.npz`,
and `surface_mapping.npz`. It refuses to overwrite them unless `--force` is
given.

Region selectors can be iterated without remeshing:

```powershell
python sim/scripts/select_fingertip_regions.py `
  --surface-stl "sim/assets/new_fingertip_custom/prepared/surface.stl" `
  --volume-msh "sim/assets/new_fingertip_custom/prepared/volume.msh" `
  --regions-config "sim/config/mechanics/regions/custom_fingertip_regions.json" `
  --output "sim/assets/new_fingertip_custom/prepared/regions.npz" --force
```

The supplied `custom.stl` currently fails before meshing: after normal STL
vertex welding it contains 204 duplicate faces and 715 non-manifold edges.
Re-export a watertight, manifold union of the **positive hollow silicone
body**. Do not export the negative manufacturing mold. The region JSON then
needs to identify the actual mount, inner coating, and touchable outer faces;
those choices are data, so no Python edit is needed.

## Run mechanics

The preserved sphere regression is the default:

```powershell
python sim/scripts/run_touch_mechanics.py --viewer null --headless
```

After preparing the custom asset:

```powershell
python sim/scripts/run_touch_mechanics.py `
  --config "sim/config/mechanics/experiments/custom_fingertip_sphere.json" `
  --viewer null --headless
```

An experiment JSON references separate asset and material JSON files. It also
contains the fingertip transform, explicit world-space contact location and
direction, indenter, approach/press/hold/release/recovery timing, optional
lateral slip, solver/contact settings, volume limits, and output rate.
Supported indenters are `sphere`, `flat_plate`, `cylinder`, and `rigid_stl`.
Rigid STL indenters additionally require `stl`, `scale_to_m`, and
`contact_point_local_m`.

Each run writes:

- `run_config.json`: fully resolved configuration, including unchanged
  material values;
- `frames.npz`: object motion, tet particles, full/inner/outer deformed
  surfaces, contact mask/area/forces/slip, displacement, tet volumes, and
  phase;
- `metrics.csv`: scalar time series;
- `surface_mapping.npz`: the immutable rest-state barycentric mapping;
- `failure_state.npz`: written only when the configured tet-volume circuit
  breaker trips.

`contact_flag` is based on positive Newton penalty reaction above the configured
force threshold. Collision-margin candidates alone do not count as contact.

## Tests

```powershell
python -m pytest -q tests/test_asset_validation.py tests/test_surface_mapping.py `
  tests/test_press_hold_release.py tests/test_contact_detection.py tests/test_repeatability.py
```
