# Tactile Tactile Sensor Simulation

## Overview
The goal of this project is to create a **realistic physics simulation of the Ecoflex silicone fingertip** used in our tactile sensor and a **virtual Time-of-Flight (ToF) depth sensor** inside it. This simulation aims to accurately reproduce the deformation of the soft fingertip during grasps to generate synthetic training data.

By combining soft-body physics with simulated depth sensing (ray casting), we aim to generate a stream of depth measurements (at 100–500 Hz) that mathematically matches real-world data.

---

## 1. Physics Simulation of the Fingertip

### Background
* **Ecoflex 00-30:** A very soft, stretchy silicone rubber.
* **Physics Engine:** NVIDIA Newton / Warp (GPU-accelerated physics engine using FEM or MPM).
* **Key Parameters:** Young's Modulus (E ≈ 30–70 kPa), Poisson's Ratio (ν ≈ 0.45–0.49), and Friction Coefficients.

### Step-by-Step Plan
1. **Set Up Environment:** Install NVIDIA Warp and Newton.
2. **Create the Fingertip Mesh:** Convert the CAD model into a tetrahedral mesh (5,000–20,000 tetrahedra) using tools like Gmsh or fTetWild.
3. **Configure Soft-Body Solver:** Load the mesh, set material parameters (e.g., E=50 kPa, ν=0.47), and anchor the base. Apply forces to verify deformation.
4. **Sim-to-Real Calibration:** Replicate real grasp scenarios from dataset. Tune parameters (Young's modulus, Poisson's ratio) to minimize MSE/maximize R² (>0.90) between simulated and real depth profiles.
5. **Document & Visualize:** Save parameters to config, record simulation video, and summarize the process.

---

## 2. Virtual Time-of-Flight (ToF) Sensor

### Background
* **Real Sensor:** A hollow Ecoflex fingertip with a reflective inner layer. An IR ToF sensor at the base measures depth changes as the silicone deforms.
* **Virtual Sensor:** Uses **ray casting** in simulation to measure distance from the sensor position to the inner silicone surface.

### Step-by-Step Plan
1. **Set Up Environment:** Familiarize with Warp's `wp.Mesh` and ray-casting API (`wp.mesh_query_ray`).
2. **Define the Virtual Sensor:** Determine Field of View (FoV), Resolution (e.g., 8x8), Sampling Rate (100-500 Hz), Position, and Orientation. Create a grid of ray directions.
3. **Implement Ray Casting:** 
   * Get the current vertex positions of the deformed mesh at each timestep.
   * Rebuild/update the `wp.Mesh` object.
   * Cast rays and record hit distances as the simulated depth array.
4. **High-Frequency Sampling Loop:** 
   * Run the simulation at the target frequency.
   * Output the depth array with a timestamp as a CSV file matching the real sensor format (e.g., `time_stamp`, `data`).
5. **(Bonus) Add Realistic Noise:** Inject Gaussian noise or simulate multipath interference.
6. **Validate Against Real Data:** Compare the simulated depth stream to real ToF readings and plot the signals.
7. **Document & Visualize:** Create a visualization (e.g., heatmap or line plot) of the depth arrays during a grasp.

---

## Deliverables Checklist
* [ ] Working tet mesh of the fingertip.
* [ ] Soft-body simulation running on GPU via Newton/Warp.
* [ ] Virtual ToF sensor (ray-casting pipeline) measuring the deforming mesh.
* [ ] High-frequency depth stream output (CSV formatting matching real data).
* [ ] Calibrated material parameters matching real-world data (Comparison plots).
* [ ] Config file with final parameters.
* [ ] Short write-up and visualizations (video/heatmap) of the process.

---

## Useful Resources
* [NVIDIA Warp Documentation](https://nvidia.github.io/warp/)
* [NVIDIA Warp — Mesh Queries & Ray Casting](https://nvidia.github.io/warp/modules/runtime.html#mesh-queries)
* [NVIDIA Newton](https://developer.nvidia.com/newton)
* [Gmsh — Mesh Generator](https://gmsh.info/)
* [Ecoflex 00-30 Datasheet](https://www.smooth-on.com/products/ecoflex-00-30/)
* Ask if you're stuck or need access to files!
