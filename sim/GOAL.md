# TouchIQ Fingertip Simulation Goal

**Status:** Project goal and acceptance document  
**Last updated:** 2026-07-23  
**Scope:** Fingertip mechanics, physical calibration, ToF coupling, and simulation data generation. This document does **not** define the STEP/STL converter website.

## 1. North-star goal

Build a reproducible, experimentally validated simulator for the TouchIQ fingertip that:

1. accepts an arbitrary **positive cured-silicone fingertip body** without source-code changes;
2. simulates its deformation, contact, recovery, and later shear/slip behavior in NVIDIA Newton;
3. produces **physically calibrated SORTA-Clear 37 force estimates**, with quantified error against real load-cell or force/torque measurements;
4. transfers the deformed sensor-facing coating surface into the validated ToF simulator;
5. reproduces the real sensor's 8×8 spatial and temporal behavior closely enough for engineering decisions and sim-to-real learning;
6. records every geometry, material, solver, calibration, and software detail needed to reproduce a run.

A visually plausible simulation is not sufficient. Quantitative claims must be supported by held-out physical experiments and reported with an error range.

## 2. Non-negotiable principles

### 2.1 Validate subsystems before coupling them

Validate the naked ToF sensor, fingertip mechanics, SORTA-Clear 37 material model, and coating/optical behavior independently. Couple them only after each subsystem has a measured baseline.

### 2.2 Use the correct geometry

The soft-body input must be the positive volume occupied by the cured silicone, not the manufacturing mold, mold halves, tooling, or an unclosed surface skin. The geometry must include the real outer contact surface, inner sensor/coating surface, wall thickness, and mounting rim.

### 2.3 Never silently repair or reinterpret an asset

Invalid topology, uncertain units, multiple ambiguous bodies, self-intersections, or implausible wall thickness must stop preparation with an actionable diagnostic. Any repair or cavity-extraction workflow must be explicit and produce a new traceable asset.

### 2.4 Calibration is batch- and assembly-specific

Datasheet values are initial estimates, not final material truth. Calibration must represent the manufactured silicone batch, coating, wall thickness, curing process, temperature range, and final mounting condition.

### 2.5 Separate physical parameters from numerical parameters

Material stiffness, compressibility, viscosity, and friction describe the real fingertip. Contact penalty stiffness, collision margins, timestep, VBD iterations, and mesh resolution are numerical settings. Numerical settings must converge and must not be tuned to hide an incorrect material model.

### 2.6 Preserve uncertainty and provenance

Every calibrated parameter set must include its source trials, fit method, validation error, operating range, software versions, mesh hashes, and uncertainty. The simulator must never label reconstructed or unvalidated forces as exact ground truth.

## 3. System boundaries

### Included

- positive silicone-body asset preparation;
- surface validation and tetrahedral meshing;
- explicit mount, inner-coating, and outer-contact regions;
- Newton soft-body deformation and rigid-object contact;
- gravity equilibration and repeatable contact trajectories;
- high-resolution surface-to-tetrahedron tracking;
- physically calibrated normal-force behavior;
- later calibration of friction, shear, slip, hysteresis, and recovery;
- offline then live coupling to the ToF simulator;
- reproducible, chunked experiment datasets;
- real-versus-sim validation and regression tests.

### Outside the current mechanics milestone

- automatic extraction of a casting cavity from arbitrary mold tooling;
- claims of validated shear or stick-slip before physical shear experiments;
- claims of exact force labels before load-cell calibration;
- replacement of the already validated naked-ToF implementation;
- robot-policy training or deployment.

## 4. Required development milestones

### Current material status

- Current mechanics material: **Smooth-On SORTA-Clear 37**
- Mechanical parameters: **provisional**
- Physical force calibration: **pending**
- Required calibration: real press-hold-release tests with synchronized load-cell and displacement data

The active starting profile is not physically calibrated and its reconstructed
reaction fields remain estimates.

## M0 — Freeze the naked ToF baseline

Validate the physical and simulated naked sensor using the same scenes:

- flat targets at multiple known distances;
- target tilt and lateral position;
- bright, dark, reflective, and low-return targets;
- moving targets at representative speeds;
- per-zone bias, variance, invalid-return rate, temporal lag, and warm-up drift.

Store a frozen baseline configuration and calibration report. Later mold-related work must not silently change this baseline.

**Gate:** naked-sensor error and noise are quantified per zone over the intended operating range.

## M1 — Fingertip asset preparation

For each fingertip design, generate and retain:

```text
surface.stl            high-resolution positive silicone boundary
volume.msh             tetrahedral Newton simulation mesh
regions.npz            mount, inner-coating, and outer-contact regions
surface_mapping.npz    surface-to-tet barycentric mapping
asset_manifest.json    units, bounds, hashes, tolerances, and provenance
```

Required validation:

- explicit source and output units;
- one intended silicone CAD solid, or an explicitly supported shell structure;
- watertight/manifold boundaries appropriate to the design;
- no duplicate or zero-area faces;
- no invalid winding or self-intersections;
- finite, nonzero enclosed silicone volume;
- physically plausible wall thickness;
- no inverted, degenerate, or unacceptable-quality tetrahedra;
- several elements through the thinnest mechanically important wall;
- correct mount and sensor-facing region selection.

**Gate:** the asset is accepted without source-code edits and its preparation report is reproducible from the original CAD export.

## M2 — Stable normal-contact mechanics

Support deterministic prescribed trajectories with:

- sphere;
- flat plate;
- cylinder;
- arbitrary rigid object STL.

Each experiment must include:

```text
settling → equilibrated baseline → approach → press → hold → release → recovery
```

Required runtime protections:

- finite state checks;
- tetrahedron compression/inversion limits;
- contact-buffer saturation detection;
- continuous indenter motion at phase boundaries;
- pinned Newton/Warp versions;
- repeatability checks within documented numerical tolerances;
- chunked failure-safe output.

**Gate:** representative normal-indentation runs complete without inversion, contact truncation, or discontinuous loading.

## M3 — Physically calibrate SORTA-Clear 37 normal forces

This milestone is mandatory before the simulator may claim physically meaningful force values.

### Physical test fixture

Use the real manufactured fingertip in its final or mechanically equivalent mount. Use:

- a calibrated load cell or wrist force/torque sensor;
- a controlled linear indenter with known geometry;
- measured displacement and synchronized timestamps;
- the same sphere/plate geometry, contact position, and trajectory used in simulation;
- recorded silicone formulation, batch, curing conditions, coating, wall thickness, mount torque or clamp condition, temperature, and trial count.

### Required experiments

Run repeated tests across:

- several indentation depths spanning the intended safe range;
- several approach and release speeds;
- press, hold, unload, and recovery phases;
- multiple contact locations;
- at least two normal indenter geometries;
- multiple repetitions to quantify manufacturing and measurement variability.

### Calibration order

1. Confirm physical geometry, wall thickness, and mounting compliance.
2. Fit hyperelastic response using force-displacement data.
3. Fit compressibility from measured lateral/inner-surface deformation where observable.
4. Fit rate dependence, relaxation, hysteresis, and recovery.
5. Verify numerical contact settings through convergence tests; do not use them as substitute material parameters.
6. Calibrate friction separately in M4.
7. Validate the current Newton reaction estimator against the load-cell result.

Start with the current Neo-Hookean model for small strain. Move to a more expressive hyperelastic and viscoelastic model—such as an Ogden/Mooney-Rivlin family with a supported relaxation model—when one parameter set cannot reproduce small- and large-strain loading, hold relaxation, unloading, and recovery.

### Fitting and validation protocol

- Divide trials into calibration, model-selection, and held-out validation sets.
- Fit one material profile per defined silicone batch/temperature regime.
- Do not retune parameters for each validation trial or each contact location.
- Report error curves, not only one aggregate score.
- Store confidence intervals or bootstrap uncertainty for fitted parameters and predictions.

### Initial engineering acceptance targets

These targets may be tightened after the measurement system's uncertainty is characterized:

- contact-onset displacement error: no greater than **0.10 mm** or the fixture resolution, whichever is larger;
- held-out normal-force RMSE: no greater than **max(0.05 N, 10% of trial peak force)**;
- held-out peak-force relative error: no greater than **10%**;
- loading/unloading hysteresis-loop area error: no greater than **15%**;
- relaxation or recovery characteristic-time error: no greater than **15%**;
- mesh refinement changes peak reaction and sensor-facing displacement by less than **5%**;
- repeated real trials and simulation predictions include reported 95% uncertainty intervals.

A force output may be called **physically calibrated** only when these or an approved replacement set of held-out criteria is met.

**Gate:** a versioned SORTA-Clear 37 material profile predicts unseen normal-indentation trials within the agreed tolerance without per-trial retuning.

## M4 — Calibrate shear, friction, and slip

After normal behavior is validated, add physical tests for:

- lateral sweeps under fixed normal indentation;
- torsion about the local surface normal;
- rolling contact;
- stick, partial slip, full slip, release, and recontact;
- several object materials and surface finishes.

The simulator must maintain path-dependent local contact state, including accumulated tangential displacement and contact break/reset events. Instantaneous tangential velocity alone must not be called a validated slip state.

**Gate:** held-out tangential-force, slip-onset, and contact-history metrics meet separately defined tolerances.

## M5 — Couple deformation to the ToF simulator

Begin with offline replay:

```text
Newton run
  → export deformed inner-coating surface at ToF timestamps
  → update the optical mesh in the frozen ToF scene
  → capture synchronized 8×8 sensor frames
```

Implement the intended sensing behavior spatially, not as a global software switch:

- where no contact-associated coating return dominates, rays may pass through the silicone and measure the object beyond it;
- where physical contact and coating geometry produce the tactile return, affected rays/zones terminate at the deformed coating;
- mixed contact must be represented per ray or per zone;
- export object path, coating path, contact mask, final selected return, intensity/confidence, and validity separately.

Do not assume this mode-selection rule is physically exact until compared with hardware. Add Snell-Descartes refraction, Fresnel losses, scattering, multipath, and coating optics when controlled experiments show that they materially affect the measured error.

**Gate:** identical real and simulated press/release trials show the same contact timing and comparable 8×8 deformation patterns within measured sensor uncertainty.

## M6 — End-to-end sim-to-real validation

Run paired real and simulated experiments with identical:

- asset geometry and mount;
- indenter/object geometry;
- contact location and orientation;
- motion trajectory;
- silicone batch and temperature profile;
- ToF configuration and sampling schedule.

Evaluate:

- force-displacement-time error;
- inner-coating displacement error;
- contact onset/release timing;
- per-zone ToF bias and variance;
- invalid-return rate;
- temporal lag and recovery;
- spatial contact-pattern similarity;
- performance on held-out objects, locations, and speeds.

**Gate:** an end-to-end validation report states the operating range, passed metrics, failed metrics, uncertainty, and known limitations.

## M7 — Dataset generation and ML readiness

After physical validation, generate randomized datasets over the validated parameter ranges only. Store simulation truth separately from sensor-like observations.

Minimum synchronized fields:

```text
time and trajectory phase
object pose, linear velocity, and angular velocity
tet particles and deformed surfaces
contact mask and approximate contact area
calibrated force estimate and uncertainty
shear/slip truth when validated
object-distance path and coating-distance path
8×8 ToF range, intensity/confidence, validity, and selected-return type
material, geometry, environment, and randomization metadata
```

Use chunked HDF5, Zarr, or manifest-indexed NPZ output. Every run must include configuration files, source and generated asset hashes, calibration profile ID, random seed, Newton/Warp/GPU environment, output schema version, and software commit.

Synthetic data may be used for training only inside the experimentally validated operating envelope, unless out-of-envelope samples are explicitly marked as exploratory or domain-randomized rather than physical ground truth.

## 5. Required quality gates

The project must maintain:

- unit tests for configuration, topology, mapping, trajectories, estimators, and exporters;
- real Newton/Warp integration tests for contact, release, repeatability, and safe failure;
- regression fixtures for valid and invalid assets;
- mesh-convergence tests;
- calibration and held-out-validation scripts;
- automatic checks that no force is labeled calibrated without a calibration profile and validation report;
- explicit schema versioning and migration rules;
- reproducible commands for every published figure or metric.

## 6. Standard outputs

Each prepared asset or simulation run should be self-contained and include, as applicable:

```text
asset_manifest.json
surface.stl
volume.msh
regions.npz
surface_mapping.npz
experiment_config.json
material_profile.json
calibration_report.json
newton_environment.json
frames_manifest.json
frames_*.npz or dataset.zarr
metrics.csv
failure_state.npz
```

## 7. Definition of done

The fingertip simulation is considered complete for engineering and sim-to-real use when all of the following are true:

1. A new positive silicone-body design can be prepared and simulated without changing Python source.
2. Geometry, units, wall thickness, region selection, tet quality, and provenance are validated.
3. Normal press-hold-release mechanics are stable and numerically converged.
4. SORTA-Clear 37 force predictions pass held-out physical calibration criteria and include uncertainty.
5. Recovery and hysteresis are reproduced over the declared operating range.
6. Shear and slip are either physically validated or explicitly excluded from claims and labels.
7. The deformed coating is coupled to the frozen ToF simulator without a global contact switch.
8. Paired real/sim ToF trials meet documented spatial and temporal error targets.
9. A complete run is reproducible from versioned assets, configurations, calibration profiles, and software commits.
10. Limitations and out-of-distribution conditions are visible in reports and datasets rather than hidden by tuning.

## 8. Immediate next actions

1. Obtain or extract the positive cured-silicone CAD body from the mold design.
2. Complete and validate the STEP/STL asset gate against the exact `prepare_fingertip.py` requirements.
3. Manufacture one calibration fingertip and build the instrumented indentation fixture.
4. Run mesh-convergence and normal-indentation calibration before coupling mechanics to ToF.
5. Preserve the naked ToF calibration as a frozen regression baseline.
