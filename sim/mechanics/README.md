# Newton fingertip mechanics

This workflow is independent of the ToF simulator. It validates and prepares a
positive hollow silicone body, maps its high-resolution surface to a tetrahedral
mesh, settles the mounted body under gravity, then runs a prescribed repeatable
normal-indentation trajectory with Newton VBD.

The current capability is a **normal indentation mechanics MVP**. Shear is not
validated, slip is not validated, and nonzero-friction/lateral-slip experiments
are provisional. Contact area, reaction, and tangential relative velocity are
estimates reconstructed from Newton's VBD penalty-contact arrays—not exact force
ground truth or a complete stick–slip model.

## Prepare a new fingertip

The input STL must be the positive cured silicone body, not the negative mold.
Copy the example regions file, update its dimensions and selectors for the new
geometry, then run:

```powershell
python sim/scripts/prepare_fingertip.py `
  --stl "D:\path\to\positive_fingertip.stl" `
  --units mm `
  --target-edge-mm 1.0 `
  --output-dir "sim/assets/my_fingertip/prepared" `
  --regions-config "sim/config/mechanics/regions/my_fingertip_regions.json"
```

Preparation creates `surface.stl`, `volume.msh`, `asset.json`, `regions.npz`,
and `surface_mapping.npz`. Add an asset JSON under
`sim/config/mechanics/assets/` and point an experiment's `asset_config` at it;
no Python edit is needed.

The former custom STL is retained only as
`tests/fixtures/invalid_two_body_mold.stl`. It has duplicate faces and
non-manifold edges and is deliberately rejected by asset preparation. The
renamed `custom_fingertip_regions.example.json` is illustrative only; regenerate
its dimensions and regions when the positive body is available.

Region selectors can be iterated without remeshing:

```powershell
python sim/scripts/select_fingertip_regions.py `
  --surface-stl "sim/assets/my_fingertip/prepared/surface.stl" `
  --volume-msh "sim/assets/my_fingertip/prepared/volume.msh" `
  --regions-config "sim/config/mechanics/regions/my_fingertip_regions.json" `
  --output "sim/assets/my_fingertip/prepared/regions.npz" --force
```

## Run mechanics

The preserved sphere experiment is the default regression:

```powershell
uv --native-tls run --project sim/newton --extra examples `
  python sim/scripts/run_touch_mechanics.py --viewer null --headless
```

Run a prepared custom asset by changing only the experiment JSON:

```powershell
uv --native-tls run --project sim/newton --extra examples `
  python sim/scripts/run_touch_mechanics.py `
  --config "sim/config/mechanics/experiments/custom_fingertip_sphere.json" `
  --viewer null --headless
```

The lifecycle is `initialization`, `settling`, `capture_baseline`, `approach`,
`press`, `hold`, `release`, `recovery`, and optional `post_recovery`. Settling
ignores fixed mount particles when checking velocity convergence. Its safe gap
is also the trajectory start gap, so the first approach substep is continuous.
Touch displacement is measured from the equilibrated baseline; CAD-rest
displacement is exported separately.

Supported indenters are `sphere`, `flat_plate`, `cylinder`, and `rigid_stl`.
Plate and cylinder support distances account for their configured orientation.

## Interactive manual deformation

This is a separate launcher; it does not execute the prescribed trajectory:

```powershell
uv --native-tls run --project sim/newton --extra examples `
  python sim/scripts/run_interactive_touch.py `
  --config "sim/config/mechanics/experiments/interactive_manual.json" `
  --viewer gl
```

Wait for the console to report that settling is complete, then use:

- **Right mouse drag:** select and move the blue probe.
- **Shift + right mouse drag:** rotate the selected probe around camera axes.
- **Space:** pause or resume physics.
- **R:** reset the probe and fingertip to the settled state.
- **B:** make the current deformation the new displacement baseline.
- **C:** save `interactive_state_XXXXX.npz` in the configured output directory.
- **Esc** or window close: exit.

The rounded-block probe starts clear of the fingertip and is the only movable
body. Mouse input changes a target transform; configurable speed and per-frame
limits move the actual kinematic probe toward it. The viewer does not apply its
picking spring. Live UI/console diagnostics report displacement, the estimated
world-space contact-force vector and magnitude, approximate area, probe pose,
minimum relative tet volume, active contacts, and contact-buffer use. These
forces remain penalty-model estimates and are not calibrated measurements.

Manual motion is transactionally safety-checked. Each candidate substep is
simulated into Newton's alternate state buffer; the controller swaps it into
the accepted state only when it passes the configured limits. Free-space,
near-contact, and contact speeds default to 25, 5, and 1.5 mm/s respectively.

The legacy asset limits **commanded indentation** to 0.75 mm. This quantity is
the probe support point commanded beyond the configured contact reference; it
is not a claim of literal rigid-object penetration. The default tet-volume
warning and recoverable-stop thresholds are `J=0.30` and `J=0.20`. The existing
`J < 0.15` fatal circuit breaker remains unchanged.

On a recoverable stop, the candidate state is discarded, the mouse target
returns to the last safe probe pose, and further inward movement is blocked.
The viewer remains open: drag outward to retract or press **R**. Diagnostics
are saved as `safety_stop_XXXXX.npz`, including the safety reason, minimum J,
affected tet IDs, candidate/last-safe probe poses, estimated force, and
commanded indentation. `display.show_safety_tets` highlights affected tet
edges; `display.show_mount_vertices` optionally shows the fixed mount region.

The default probe is `sim/assets/probes/rounded_block.stl`. The same `probe`
section also accepts `capsule`, `cylinder`, `sphere`, or
`custom_rigid_stl`. The interactive configuration reuses the normal
asset/material/solver/contact/equilibration schema; its generated compatibility
trajectory is used only while building the shared Newton model and is never
advanced by the interactive controller.

## Record an MP4

The `video` section is optional and disabled by default. To record through
recovery and an optional post-recovery tail:

```powershell
uv --native-tls run --project sim/newton --extra examples `
  --with imageio --with imageio-ffmpeg `
  python sim/scripts/run_touch_mechanics.py `
  --config "sim/config/mechanics/experiments/sphere_visible_demo.json" `
  --viewer gl --headless --record-video
```

Recording requires the GL viewer. Omit `--headless` for a visible window, and
use `--video-path` to override the filename. `trajectory.post_recovery_s`
controls the recovered-pose tail and is always simulated whether video is on
or off, so recording cannot change the mechanics experiment.

## Newton compatibility and repeatability

The supported runtime is Newton `1.2.0.dev0` at Git SHA
`8baee876dc5f001c66f1cbafec16246a3fb6f6f6`. Every run records the active SHA
and warns if it differs; pass `--strict-newton-version` or set
`solver.newton_strict` to fail instead. `contact.py` depends on internal Newton
contact arrays and must be revalidated after a Newton update.

`solver.deterministic` requests deterministic execution options when the active
Newton API exposes them. The pinned API does not currently expose such options
for VBD/collision construction, so only the prescribed trajectory is
deterministic. Repeatability is checked numerically, not claimed bitwise.

## Output schema 2

Every output directory is self-contained:

- `run_config.json`, `asset_manifest.json`, `regions.npz`,
  `surface_mapping.npz`, and `newton_environment.json` describe the run;
- `frames_00000.npz`, `frames_00001.npz`, and so on contain bounded chunks;
- `frames_manifest.json` lists and versions the chunks;
- `metrics.csv` is streamed incrementally;
- `failure_state.npz` is written for a controlled volume, equilibration, or
  contact-buffer failure.

Canonical contact fields are `approx_contact_area_m2`,
`estimated_axial_reaction_n`, `estimated_transverse_reaction_n`, and
`estimated_tangential_relative_velocity_m_s`. Schema version 2 retains the old
field names as deprecated aliases for one transition version.

The configured, current, and maximum VBD rigid-particle contact counts are
exported with saturation status and first-saturation indices. A count at or
above `rigid_body_particle_contact_buffer_size` stops the run instead of
silently accepting dropped contacts.
Object velocity is split into `object_linear_velocity_m_s` and
`object_angular_velocity_rad_s`.

## Tests

Run the mechanics unit tests (including the invalid-STL fixture):

```powershell
uv --native-tls run --project sim/newton --with pytest --with trimesh `
  python -m pytest -o addopts= -q `
  tests/test_asset_validation.py tests/test_example_configs.py `
  tests/test_indenter_orientation.py tests/test_surface_mapping.py `
  tests/test_press_hold_release.py tests/test_contact_detection.py `
  tests/test_repeatability.py tests/test_video_recording.py
```

The `newton_integration` tests are intentionally excluded from plain `pytest`
by `pytest.ini`; run the real rollout, saturation, and numerical repeatability
tests explicitly on the target Newton/Warp/GPU environment:

```powershell
uv --native-tls run --project sim/newton --extra examples --with pytest `
  python -m pytest -o addopts= -m newton_integration tests/integration -q
```
