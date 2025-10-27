# Hydraulic Erosion MPI Implementation

This directory contains my attempt to parallelize the hydraulic erosion model using MPI for domain decomposition.

## Project Overview

A 2D hydraulic erosion simulation that models water flow across terrain using the "virtual pipes" method. The project includes both a serial reference implementation and a parallel MPI version.

## Files

- **[benchmark_scaling.py](benchmark_scaling.py)** - Performance benchmarking script for scaling analysis.
- **[erosion_serial.py](erosion_serial.py)** - Serial NumPy implementation, extracted exactly from the original Jupyter notebook. Serves as our ground truth reference.
- **[erosion_mpi.py](erosion_mpi.py)** - Parallel MPI version attempting to use 1D domain decomposition with halo exchange.
- **[visualize_results.py](visualize_results.py)** - Creates gif and mp4 animations from simulation output.
- **[visualize_results_with_terrain.py](visualize_results_with_terrain.py)** - Same as above, but adds a terrain visual on the left side.

## Current Status: Known Issues

⚠️ **The MPI implementation currently produces visible artifacts at rank boundaries.** When running with multiple processes, visualizations show clear horizontal discontinuities where ranks meet, indicating that information is not being correctly exchanged between processes.

## Key MPI Concepts Implemented

The [erosion_mpi.py](erosion_mpi.py) file is structured as:

1. **Domain decomposition setup** - The grid is split horizontally into strips:

   ```
   Original grid (8 rows):     After decomposition (4 processes):
   ┌─────────────┐             Rank 0: ┌─────────────┐
   │ row 0       │                     │ row 0       │
   │ row 1       │                     │ row 1       │
   │ row 2       │             ────────┼─────────────┤ ← Boundary between ranks
   │ row 3       │             Rank 1: │ row 2       │
   │ row 4       │                     │ row 3       │
   │ row 5       │             ────────┼─────────────┤ ← Boundary between ranks
   │ row 6       │             Rank 2: │ row 4       │
   │ row 7       │                     │ row 5       │
   └─────────────┘             ────────┼─────────────┤ ← Boundary between ranks
                               Rank 3: │ row 6       │
                                       │ row 7       │
                                       └─────────────┘
   ```
2. **Halo exchange functions** - The critical MPI communication
   ```
   Rank 1's local array with ghost cells:
   ┌──────────────┐  ← Ghost row (should receive from Rank 0's bottom)
   ├──────────────┤
   │ row 2        │  ← Real data owned by Rank 1
   │ row 3        │  ← Real data owned by Rank 1
   ├──────────────┤
   └──────────────┘  ← Ghost row (should receive from Rank 2's top)
   ```
- ⚠️ **This is likely where the bug is** - the boundary artifacts suggest this isn't working correctly.

3. **Flow update** - Physics calculations (mostly identical to serial version), except:
- We first call exchange_halo() to get neighbor data
- We apply boundary conditions only on physical domain edges (not between ranks)

4. **Gathering results** - Periodically gather data from all ranks and save snapshots for animation.

5. **Main simulation loop** - Time stepping with periodic output.

## Quick Start

### Go into the `hydraulic_erosion` directory for paths to work

```bash
# Clone this repo and cd into for script paths to work properly
git clone https://github.com/tedtop/CSCI-491-591-HPC.git
cd hydraulic_erosion
```

### Performance benchmarking (will optionally generate animations)

```bash
# Run performance benchmarking
python benchmark_scaling.py
```

### Create individual animations

```bash
# Create individual animations
python visualize_results.py output/erosion_mpi_4procs_256x256.npz

# Create individual animations (with output directory)
python visualize_results.py output/erosion_mpi_4procs_256x256.npz --output-dir animations

# Create individual animations with terrain
python visualize_results_with_terrain.py output/erosion_mpi_4procs_256x256.npz
```

### Compare serial and MPI (.npz) outputs
```bash
# Compare .npz output files from serial and MPI implementations
python run_and_compare.py --grid-size 512 --nprocs 4
```

## Sources
- Hydraulic Erosion Notebook –  https://colab.research.google.com/github/JacobDowns/CSCI-491-591/blob/main/hydraulic_erosion_demo.ipynb
- Xing Mei, Philippe Decaudin, Bao-Gang Hu. Fast Hydraulic Erosion Simulation and Visualization on GPU. PG ’07 - 15th Pacific Conference on Computer Graphics and Applications, Oct 2007, Maui, United States. pp.47-56, ff10.1109/PG.2007.15ff. ffinria-00402079f – https://inria.hal.science/inria-00402079/document
- AI Attribution: Claude to help with numerical comparison between serial & MPI output, troubleshooting halo exchange, generating benchmark plots.