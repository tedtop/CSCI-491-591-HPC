"""
Run and Compare - Automated Testing Script
===========================================

This script:
1. Runs the serial version
2. Runs the MPI version
3. Loads both output .npz files
4. Compares them in detail

Usage:
    # Compare serial vs MPI with 4 processes, 256x256 grid
    python run_and_compare.py

    # Custom grid size and process count
    python run_and_compare.py --grid-size 512 --nprocs 4

    # Quick test with small grid
    python run_and_compare.py --grid-size 128 --nprocs 2
"""

import subprocess
import numpy as np
import sys
import argparse
from pathlib import Path
import time


def run_serial(grid_size):
    """
    Run the serial version

    Parameters
    ----------
    grid_size : int
        Grid dimension

    Returns
    -------
    output_file : Path
        Path to output .npz file
    elapsed : float
        Time taken in seconds
    """
    print("="*70)
    print("STEP 1: Running Serial Version")
    print("="*70)

    output_file = Path("output") / f"erosion_serial_{grid_size}x{grid_size}.npz"

    # Delete old output if exists
    if output_file.exists():
        output_file.unlink()
        print(f"Deleted old output: {output_file}")

    cmd = ["python", "erosion_serial.py", str(grid_size)]
    print(f"Command: {' '.join(cmd)}")
    print()

    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"ERROR: Serial version failed with code {result.returncode}")
        sys.exit(1)

    if not output_file.exists():
        print(f"ERROR: Expected output file not found: {output_file}")
        sys.exit(1)

    print()
    print(f"✓ Serial version complete in {elapsed:.2f}s")
    print(f"✓ Output: {output_file}")
    print()

    return output_file, elapsed


def run_mpi(grid_size, nprocs):
    """
    Run the MPI version

    Parameters
    ----------
    grid_size : int
        Grid dimension
    nprocs : int
        Number of MPI processes

    Returns
    -------
    output_file : Path
        Path to output .npz file
    elapsed : float
        Time taken in seconds
    """
    print("="*70)
    print("STEP 2: Running MPI Version")
    print("="*70)

    output_file = Path("output") / f"erosion_mpi_{nprocs}procs_{grid_size}x{grid_size}.npz"

    # Delete old output if exists
    if output_file.exists():
        output_file.unlink()
        print(f"Deleted old output: {output_file}")

    cmd = ["mpirun", "-n", str(nprocs), "python", "erosion_mpi.py", str(grid_size)]
    print(f"Command: {' '.join(cmd)}")
    print()

    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"ERROR: MPI version failed with code {result.returncode}")
        sys.exit(1)

    if not output_file.exists():
        print(f"ERROR: Expected output file not found: {output_file}")
        sys.exit(1)

    print()
    print(f"✓ MPI version complete in {elapsed:.2f}s")
    print(f"✓ Output: {output_file}")
    print()

    return output_file, elapsed


def compare_npz_files(serial_file, mpi_file):
    """
    Load and compare two .npz files

    Parameters
    ----------
    serial_file : Path
        Serial output file
    mpi_file : Path
        MPI output file

    Returns
    -------
    comparison : dict
        Comparison results
    """
    print("="*70)
    print("STEP 3: Comparing Output Files")
    print("="*70)
    print()

    # Load files
    print(f"Loading serial: {serial_file}")
    serial_data = np.load(serial_file)

    print(f"Loading MPI:    {mpi_file}")
    mpi_data = np.load(mpi_file)
    print()

    # Check what's in each file
    print("Serial file contains:")
    for key in serial_data.files:
        shape = serial_data[key].shape if hasattr(serial_data[key], 'shape') else 'scalar'
        print(f"  {key}: {shape}")
    print()

    print("MPI file contains:")
    for key in mpi_data.files:
        shape = mpi_data[key].shape if hasattr(mpi_data[key], 'shape') else 'scalar'
        print(f"  {key}: {shape}")
    print()

    # Extract data
    times_serial = serial_data['times']
    H_serial = serial_data['H_hist']
    z_serial = serial_data['z']

    times_mpi = mpi_data['times']
    H_mpi = mpi_data['H_hist']
    z_mpi = mpi_data['z']

    # Compare metadata
    print("-"*70)
    print("METADATA COMPARISON")
    print("-"*70)

    metadata_match = True

    # Number of snapshots
    n_serial = len(times_serial)
    n_mpi = len(times_mpi)
    if n_serial == n_mpi:
        print(f"✓ Number of snapshots: {n_serial}")
    else:
        print(f"✗ Number of snapshots differ: {n_serial} vs {n_mpi}")
        metadata_match = False

    # Grid size
    if z_serial.shape == z_mpi.shape:
        print(f"✓ Grid size: {z_serial.shape}")
    else:
        print(f"✗ Grid sizes differ: {z_serial.shape} vs {z_mpi.shape}")
        metadata_match = False

    # Time arrays
    if n_serial == n_mpi and np.allclose(times_serial, times_mpi):
        print(f"✓ Time arrays match")
    elif n_serial != n_mpi:
        print(f"✗ Time arrays have different lengths")
        metadata_match = False
    else:
        max_time_diff = np.max(np.abs(times_serial - times_mpi))
        print(f"✗ Time arrays differ by max {max_time_diff:.6f}")
        metadata_match = False

    print()

    # Compare terrain
    print("-"*70)
    print("TERRAIN COMPARISON")
    print("-"*70)

    z_diff = np.abs(z_serial - z_mpi)
    z_max_diff = z_diff.max()
    z_mean_diff = z_diff.mean()

    print(f"Terrain max difference:  {z_max_diff:.6f}")
    print(f"Terrain mean difference: {z_mean_diff:.6f}")

    if z_max_diff < 1e-10:
        print("✓ Terrain matches perfectly")
        terrain_match = True
    else:
        print("✗ Terrain differs (should be identical!)")
        terrain_match = False

    print()

    # Compare water heights
    print("-"*70)
    print("WATER HEIGHT COMPARISON")
    print("-"*70)

    if n_serial != n_mpi:
        print("Cannot compare snapshots - different counts")
        return {
            'metadata_match': metadata_match,
            'terrain_match': terrain_match,
            'water_match': False,
            'max_error': float('inf'),
            'mean_error': float('inf')
        }

    max_errors = []
    mean_errors = []

    print(f"Comparing {n_serial} snapshots...")
    print()
    print("Snapshot    Time(s)    Max Error    Mean Error")
    print("-"*70)

    for i in range(n_serial):
        h_diff = np.abs(H_serial[i] - H_mpi[i])
        max_err = h_diff.max()
        mean_err = h_diff.mean()

        max_errors.append(max_err)
        mean_errors.append(mean_err)

        # Print every 10th or first/last
        if i % 10 == 0 or i == n_serial - 1:
            # Use fixed-point formatting to avoid scientific notation for large errors
            print(f"{i:8d}    {times_serial[i]:6.2f}    {max_err:10.6f}    {mean_err:10.6f}")

    print()
    print("-"*70)

    overall_max = np.max(max_errors)
    overall_mean = np.mean(mean_errors)

    print(f"Overall Maximum Error: {overall_max:.6f}")
    print(f"Overall Mean Error:    {overall_mean:.6f}")
    print()

    # Determine verdict
    if overall_max < 1e-10:
        verdict = "PERFECT MATCH"
        symbol = "✓✓✓"
        water_match = True
    elif overall_max < 1e-6:
        verdict = "EXCELLENT (within numerical precision)"
        symbol = "✓✓"
        water_match = True
    elif overall_max < 1e-3:
        verdict = "GOOD (small differences)"
        symbol = "✓"
        water_match = True
    elif overall_max < 0.1:
        verdict = "FAIR (noticeable differences)"
        symbol = "~"
        water_match = False
    else:
        verdict = "POOR (large differences)"
        symbol = "✗"
        water_match = False

    print(f"{symbol} {verdict}")
    print()

    return {
        'metadata_match': metadata_match,
        'terrain_match': terrain_match,
        'water_match': water_match,
        'max_error': overall_max,
        'mean_error': overall_mean,
        'verdict': verdict
    }


def main():
    parser = argparse.ArgumentParser(description='Run and compare serial vs MPI versions')
    parser.add_argument('--grid-size', type=int, default=256,
                        help='Grid size (default: 256)')
    parser.add_argument('--nprocs', type=int, default=4,
                        help='Number of MPI processes (default: 4)')

    args = parser.parse_args()

    print()
    print("="*70)
    print("HYDRAULIC EROSION - SERIAL vs MPI COMPARISON")
    print("="*70)
    print(f"Grid size: {args.grid_size} × {args.grid_size}")
    print(f"MPI processes: {args.nprocs}")
    print()

    # Run serial version
    serial_file, serial_time = run_serial(args.grid_size)

    # Run MPI version
    mpi_file, mpi_time = run_mpi(args.grid_size, args.nprocs)

    # Compare outputs
    comparison = compare_npz_files(serial_file, mpi_file)

    # Final summary
    print("="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print()
    print("Performance:")
    print(f"  Serial time:  {serial_time:6.2f} s")
    print(f"  MPI time:     {mpi_time:6.2f} s")
    if mpi_time > 0:
        speedup = serial_time / mpi_time
        print(f"  Speedup:      {speedup:6.2f} x")
    print()

    print("Correctness:")
    print(f"  Metadata match:  {'✓' if comparison['metadata_match'] else '✗'}")
    print(f"  Terrain match:   {'✓' if comparison['terrain_match'] else '✗'}")
    print(f"  Water match:     {'✓' if comparison['water_match'] else '✗'}")
    print()
    print(f"  Max error:       {comparison['max_error']:.6f}")
    print(f"  Mean error:      {comparison['mean_error']:.6f}")
    print(f"  Verdict:         {comparison['verdict']}")
    print()

    print("Output files:")
    print(f"  Serial:  {serial_file}")
    print(f"  MPI:     {mpi_file}")
    print()

    print("Next steps:")
    print(f"  # Visualize serial results")
    print(f"  python visualize_results.py {serial_file}")
    print()
    print(f"  # Visualize MPI results")
    print(f"  python visualize_results.py {mpi_file}")
    print()
    print("="*70)


if __name__ == "__main__":
    main()
