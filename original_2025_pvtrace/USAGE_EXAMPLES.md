# GPU Ray-Tracing Simulation — Usage Guide

A Monte Carlo ray-tracing library for computing **Light Extraction Efficiency (LEE)** of
hemispherical dome and rectangular slab optical geometries. Simulations run on the GPU
(via OpenCL / PyOpenCL) and fall back transparently to a multi-threaded CPU path when no
GPU is available.

---

## Requirements

```
pvtrace
numpy
pyopencl       # optional — enables GPU acceleration
pandas
matplotlib
```

Install Python dependencies:

```bash
pip install pvtrace pyopencl pandas matplotlib
```

---

## GPU Setup (AMD / Mesa on Fedora/Linux)

The simulation uses the Mesa Rusticl OpenCL driver for AMD integrated and discrete GPUs.
Two steps are required the first time:

### 1. Install the Mesa OpenCL driver (requires root, one-time only)

```bash
sudo dnf install -y mesa-libOpenCL
```

### 2. Enable the Rusticl driver for your GPU

Mesa Rusticl requires an environment variable to activate for specific GPU families.
For AMD Radeon (RDNA/GCN — the `radeonsi` driver):

```bash
export RUSTICL_ENABLE=radeonsi
```

To make this permanent so every terminal session has GPU acceleration automatically:

```bash
echo 'export RUSTICL_ENABLE=radeonsi' >> ~/.bashrc
source ~/.bashrc
```

### Verify GPU is detected

```bash
python3 -c "
import pyopencl as cl
for p in cl.get_platforms():
    print(p.name, [d.name for d in p.get_devices()])
"
```

Expected output (AMD Radeon 860M example):
```
rusticl ['AMD Radeon 860M Graphics (radeonsi, gfx1152, LLVM 21.1.8, ...)']
```

If you see `rusticl []` (platform found but no devices listed), `RUSTICL_ENABLE=radeonsi`
is not set in the current shell.

---

## Core Concepts

| Term | Definition |
|---|---|
| **LEE** | Light Extraction Efficiency — fraction of emitted photons that escape the geometry |
| **Dome** | Hemispherical glass lens placed over a circular emitter |
| **Slab** | Rectangular glass waveguide with a square emitter on the bottom face |
| **Escaped** | Ray that exits the geometry into the surrounding medium |
| **Absorbed** | Ray captured by the emitter body or by bulk/luminophore absorption |
| **Trapped** | Ray that exceeds `max_steps` without resolving — excluded from LEE |

LEE is defined as:

```
LEE = escaped_count / (escaped_count + absorbed_count)
```

Trapped rays are excluded from the denominator. Typical LEE values range from ~0.4 to ~0.99
depending on geometry and material properties.

---

## Running the Demo Scripts

The fastest way to get started is to run the included demos. With GPU enabled:

```bash
RUSTICL_ENABLE=radeonsi python3 demo_slab.py
RUSTICL_ENABLE=radeonsi python3 demo_dome.py
```

Or, if `RUSTICL_ENABLE` is already in your environment:

```bash
python3 demo_slab.py
python3 demo_dome.py
```

Each demo runs four analyses (thickness/radius sweep, surface scatter sweep, multi-RI
comparison, and 2D optimization) and saves PNG plots to the working directory.

**Expected output — slab demo:**
```
Slab Parameter Sweep Demo
==============================
1. Thickness sweep:
   Best: thickness=10.0, LEE=0.991
2. Surface roughness sweep:
   Best: roughness=0.0, LEE=0.842
3. RI comparison:
   Comparison plot saved
4. 2D optimization:
   Global optimum: thickness=3.0, RI=1.5, LEE=0.978

Files created:
  slab_thickness.png
  slab_roughness.png
  slab_ri_comparison.png
  slab_2d_optimization.png
```

---

## 1. Running a Single Simulation

### Slab geometry

```python
from simulation_physics import create_slab_config, UniversalSimulator, calculate_lee

config = create_slab_config(
    slab_side_length=10.0,          # mm — lateral size of the square slab
    slab_thickness=3.0,             # mm — vertical thickness
    emitter_side_length=1.0,        # mm — square emitter footprint
    emitter_thickness=0.1,          # mm — emitter body height
    glass_refractive_index=1.5,     # dimensionless
    slab_absorption_coefficient=0.0,# mm⁻¹ — bulk glass loss
    top_surface_roughness=0.0,      # 0–1 — Lambertian scatter probability at top face
    luminophore_absorption_coefficient=0.0,
    luminophore_quantum_yield=1.0,
    ray_count=10000,
    max_steps=1000,
    use_gpu=True                    # Falls back to CPU automatically if GPU unavailable
)

simulator = UniversalSimulator(config)
result = simulator.run_simulation()

lee = calculate_lee(result)
print(f"LEE:      {lee:.4f}")
print(f"Escaped:  {result.escaped_count}")
print(f"Absorbed: {result.absorbed_count}")
print(f"Trapped:  {result.trapped_count}")
print(f"Elapsed:  {result.execution_time:.2f} s")
```

### Dome geometry

```python
from simulation_physics import create_dome_config, UniversalSimulator, calculate_lee

config = create_dome_config(
    dome_radius=5.0,                # mm
    emitter_radius=1.0,             # mm — circular emitter radius
    emitter_height=0.1,             # mm — emitter cylinder height
    glass_refractive_index=1.8,
    dome_absorption_coefficient=0.0,
    scatter_probability=0.0,        # 0–1 — Lambertian scatter at curved surface
    luminophore_absorption_coefficient=0.0,
    luminophore_quantum_yield=1.0,
    ray_count=10000,
    use_gpu=True
)

simulator = UniversalSimulator(config)
result = simulator.run_simulation()
print(f"LEE: {calculate_lee(result):.4f}")
```

---

## 2. Convenience One-Liners

```python
from simulation_physics import run_slab_simulation, run_dome_simulation, calculate_lee

# All keyword arguments are forwarded to create_slab_config / create_dome_config
result = run_slab_simulation(slab_thickness=2.0, glass_refractive_index=1.8, ray_count=5000)
print(f"Slab LEE: {calculate_lee(result):.4f}")

result = run_dome_simulation(dome_radius=4.0, scatter_probability=0.3, ray_count=5000)
print(f"Dome LEE: {calculate_lee(result):.4f}")
```

---

## 3. Fluent Configuration Builder

`ConfigurationBuilder` provides a chainable API for constructing configurations:

```python
from simulation_physics import ConfigurationBuilder, UniversalSimulator, calculate_lee

config = (
    ConfigurationBuilder()
    .slab(side_length=10.0, thickness=2.0, emitter_side_length=1.0, emitter_thickness=0.1)
    .glass(refractive_index=2.0, absorption=0.01)
    .surface(roughness=0.3)
    .luminophore(absorption=0.5, quantum_yield=0.9)
    .simulation(ray_count=5000, max_steps=500)
    .build()
)

result = UniversalSimulator(config).run_simulation()
print(f"LEE: {calculate_lee(result):.4f}")
```

Dome variant:

```python
config = (
    ConfigurationBuilder()
    .dome(radius=5.0, emitter_radius=1.0, emitter_height=0.1)
    .glass(refractive_index=1.8)
    .surface(scatter_probability=0.5)
    .simulation(ray_count=5000)
    .build()
)
```

**Builder method reference:**

| Method | Parameters | Notes |
|---|---|---|
| `.dome(radius, emitter_radius, emitter_height)` | mm | Sets geometry to dome |
| `.slab(side_length, thickness, emitter_side_length, emitter_thickness)` | mm | Sets geometry to slab |
| `.glass(refractive_index, absorption)` | — / mm⁻¹ | Sets material RI and bulk absorption |
| `.surface(scatter_probability, roughness)` | 0–1 | `scatter_probability` for dome; `roughness` for slab |
| `.luminophore(absorption, quantum_yield)` | mm⁻¹ / 0–1 | Optional wavelength-shifting component |
| `.emitter(absorption)` | mm⁻¹ | Emitter body absorptivity (default 1000) |
| `.simulation(ray_count, max_steps)` | — | Number of photons and step limit |
| `.build()` | — | Returns `SimulationConfig` |

---

## 4. Parameter Sweeps

### 4a. 1-D sweep

```python
from simulation_physics import create_slab_config
from parameter_sweep import GPUParameterSweep, SweepPlotter

base_config = create_slab_config(ray_count=3000, use_gpu=True)
sweep = GPUParameterSweep(base_config)
plotter = SweepPlotter()

# Returns (list_of_values, list_of_LEE)
param_values, lee_values = sweep.sweep_1d(
    'slab_thickness',
    [0.5, 1.0, 2.0, 3.0, 4.0],
    parallel=True,      # use ThreadPoolExecutor (default)
    max_workers=4,
    save_csv='thickness_sweep.csv'   # optional; writes detailed ray stats
)

plotter.plot_1d(param_values, lee_values, 'slab_thickness', save_path='thickness.png')
```

### 4b. 2-D optimization sweep

```python
import numpy as np

t_values  = [1.0, 2.0, 3.0, 4.0]
ri_values = [1.5, 2.0, 2.5]

# Returns three numpy arrays of shape (len(t_values), len(ri_values))
t_grid, ri_grid, lee_grid = sweep.sweep_2d(
    'slab_thickness', t_values,
    'glass_refractive_index', ri_values,
    save_csv='2d_results.csv'
)

plotter.plot_2d(t_grid, ri_grid, lee_grid,
                'slab_thickness', 'glass_refractive_index',
                save_path='optimization.png')

max_idx = np.unravel_index(np.argmax(lee_grid), lee_grid.shape)
print(f"Optimum: thickness={t_grid[max_idx]:.1f}, RI={ri_grid[max_idx]:.1f}, LEE={lee_grid[max_idx]:.4f}")
```

### 4c. Custom parameter sets

```python
parameter_sets = [
    {'slab_thickness': 1.0, 'glass_refractive_index': 1.5, 'top_surface_roughness': 0.0},
    {'slab_thickness': 2.0, 'glass_refractive_index': 2.0, 'top_surface_roughness': 0.5},
    {'slab_thickness': 3.0, 'glass_refractive_index': 2.5, 'top_surface_roughness': 1.0},
]

results = sweep.sweep_custom(parameter_sets, save_csv='custom.csv')
for params, lee in results:
    print(f"LEE={lee:.4f}  {params}")
```

### 4d. Comparing multiple configurations

```python
ri_values = [1.5, 2.0, 2.5]
data_sets = {}

for ri in ri_values:
    cfg = create_slab_config(glass_refractive_index=ri, ray_count=2000)
    ri_sweep = GPUParameterSweep(cfg)
    _, lee_values = ri_sweep.sweep_1d('slab_thickness', [1.0, 2.0, 3.0, 4.0])
    data_sets[f'RI={ri}'] = ([1.0, 2.0, 3.0, 4.0], lee_values)

plotter.plot_comparison(data_sets, 'slab_thickness', save_path='ri_comparison.png')
```

---

## 5. Individual Sweep Examples (`parameter_sweep.py`)

```bash
python3 parameter_sweep.py thickness   # 1D slab-thickness sweep + CSV export
python3 parameter_sweep.py ri          # RI comparison across multiple thicknesses
python3 parameter_sweep.py 2d          # 2D thickness × RI optimization + CSV export
python3 parameter_sweep.py dome        # 1D dome-radius sweep
python3 parameter_sweep.py custom      # Custom multi-parameter combinations
```

---

## 6. All Sweepable Parameters

### Slab geometry

| Parameter | Type | Unit | Description |
|---|---|---|---|
| `slab_thickness` | float | mm | Vertical thickness of the glass slab |
| `slab_side_length` | float | mm | Lateral size (square footprint) |
| `emitter_side_length` | float | mm | Square emitter footprint |
| `emitter_thickness` | float | mm | Emitter body height |
| `glass_refractive_index` | float | — | Refractive index of the glass |
| `slab_absorption_coefficient` | float | mm⁻¹ | Bulk absorption in glass |
| `top_surface_roughness` | float | 0–1 | Lambertian scatter fraction at top face |
| `luminophore_absorption_coefficient` | float | mm⁻¹ | Luminophore absorption |
| `luminophore_quantum_yield` | float | 0–1 | Probability of re-emission after absorption |
| `ray_count` | int | — | Number of photons per simulation |

### Dome geometry

| Parameter | Type | Unit | Description |
|---|---|---|---|
| `dome_radius` | float | mm | Hemisphere radius |
| `emitter_radius` | float | mm | Circular emitter radius |
| `emitter_height` | float | mm | Emitter cylinder height |
| `glass_refractive_index` | float | — | Refractive index of the dome glass |
| `dome_absorption_coefficient` | float | mm⁻¹ | Bulk absorption in dome glass |
| `scatter_probability` | float | 0–1 | Lambertian scatter probability at curved surface |
| `luminophore_absorption_coefficient` | float | mm⁻¹ | Luminophore absorption |
| `luminophore_quantum_yield` | float | 0–1 | Quantum yield of re-emission |
| `ray_count` | int | — | Number of photons per simulation |

---

## 7. SimulationResults Fields

`UniversalSimulator.run_simulation()` returns a `SimulationResults` dataclass:

| Field | Type | Description |
|---|---|---|
| `total_rays` | int | Total rays emitted |
| `escaped_count` | int | Rays that left the geometry |
| `absorbed_count` | int | Rays absorbed (bulk, luminophore, or emitter) |
| `trapped_count` | int | Rays that hit `max_steps` without resolving |
| `error_count` | int | Rays that raised an exception during tracing |
| `exit_angles` | `List[float]` | Exit angles in degrees (CPU path only) |
| `path_lengths` | `Dict[str, List[float]]` | Per-outcome path lengths in mm (CPU path only) |
| `bounce_counts` | `Dict[str, List[int]]` | Per-outcome surface bounce counts (CPU path only) |
| `interaction_stats` | `Dict[str, int]` | Surface interaction counters (TIR, scatter, etc.) |
| `execution_time` | float | Wall-clock time in seconds |
| `bulk_absorption_count` | int | Rays absorbed by bulk glass |
| `emitter_absorption_count` | int | Rays re-absorbed by the emitter body |
| `luminophore_absorption_count` | int | Rays absorbed by the luminophore |

---

## 8. GPU vs CPU Behaviour

- GPU acceleration is enabled by default (`use_gpu=True`) and requires `pyopencl` and a
  working OpenCL ICD (see GPU Setup above).
- If no OpenCL device is found, or `pyopencl` is not installed, the simulator falls back
  silently to the CPU path with no code changes required.
- On CPU, rays are processed in parallel using `ProcessPoolExecutor` for batches > 1000
  rays, or `ThreadPoolExecutor` for smaller batches.
- `exit_angles`, `path_lengths`, and `bounce_counts` are populated on the CPU path only;
  they are empty dicts/lists for GPU-accelerated runs. LEE and absorption counts are
  always available regardless of backend.
- The GPU threshold is 10 rays for dome geometry and 100 rays for slab geometry.
  Batches below these thresholds fall through to the CPU path automatically.

---

## 9. Known Warnings

**`RepeatedKernelRetrieval`** — emitted by pyopencl during sweeps:
```
RepeatedKernelRetrieval: Kernel 'trace_slab_rays' has been retrieved more than once.
```
This is harmless. It appears because the OpenCL kernel is looked up by name on each
simulation call. It does not affect correctness or results, only produces a log message.
To suppress it entirely, set the logging level before running:

```python
import logging
logging.getLogger('pyopencl').setLevel(logging.WARNING)
```

**Matplotlib `DEBUG` font messages** — these appear when matplotlib is scanning system
fonts and are entirely normal. They can be suppressed with:

```python
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
```