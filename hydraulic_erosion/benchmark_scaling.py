"""
Benchmark Scaling Performance
==============================

This script runs the hydraulic erosion simulation with different grid sizes
and number of MPI processes, then creates performance scaling plots.

Usage:
    python benchmark_scaling.py
"""

import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json
import sys
import shutil


def run_serial(grid_size):
    """
    Run serial version and measure time

    Returns:
    --------
    elapsed : float
        Wall-clock time in seconds
    """
    print(f"  Running serial with grid {grid_size}x{grid_size}...", end=" ", flush=True)

    cmd = ["python", "erosion_serial.py", str(grid_size)]
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"FAILED")
        print(result.stderr)
        return None

    print(f"{elapsed:.2f}s")
    return elapsed


def run_mpi(grid_size, nprocs):
    """
    Run MPI version and measure time

    Returns:
    --------
    elapsed : float
        Wall-clock time in seconds
    """
    print(f"  Running MPI with {nprocs} procs, grid {grid_size}x{grid_size}...", end=" ", flush=True)

    cmd = ["mpirun", "-n", str(nprocs), "python", "erosion_mpi.py", str(grid_size)]
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"FAILED")
        print(result.stderr)
        return None

    print(f"{elapsed:.2f}s")
    return elapsed


def run_benchmarks():
    """
    Run all benchmark combinations

    Returns:
    --------
    results : dict
        Nested dictionary with timing results
    """
    grid_sizes = [256, 512, 1024, 2048]
    proc_counts = [1, 2, 4, 8]

    results = {
        'grid_sizes': grid_sizes,
        'proc_counts': proc_counts,
        'serial_times': {},
        'mpi_times': {},
        'speedups': {},
    }

    print("="*70)
    print("PERFORMANCE BENCHMARKING")
    print("="*70)
    print()

    # Run serial benchmarks
    print("Serial Benchmarks:")
    print("-"*70)
    for grid_size in grid_sizes:
        elapsed = run_serial(grid_size)
        if elapsed is not None:
            results['serial_times'][grid_size] = elapsed
    print()

    # Run MPI benchmarks
    print("MPI Benchmarks:")
    print("-"*70)
    for grid_size in grid_sizes:
        results['mpi_times'][grid_size] = {}
        results['speedups'][grid_size] = {}

        for nprocs in proc_counts:
            elapsed = run_mpi(grid_size, nprocs)
            if elapsed is not None:
                results['mpi_times'][grid_size][nprocs] = elapsed

                # Calculate speedup relative to serial
                if grid_size in results['serial_times']:
                    speedup = results['serial_times'][grid_size] / elapsed
                    results['speedups'][grid_size][nprocs] = speedup

        print()

    return results


def plot_results(results):
    """
    Create visualization plots

    Parameters:
    -----------
    results : dict
        Benchmark results
    """
    grid_sizes = results['grid_sizes']
    proc_counts = results['proc_counts']

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # ========================================
    # Figure 1: Execution Time vs Processes
    # ========================================
    print("Creating benchmark scaling plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1a: Absolute times
    for grid_size in grid_sizes:
        if grid_size in results['mpi_times']:
            procs = []
            times = []

            # Add serial time as first point (labeled as 0 procs for visual)
            if grid_size in results['serial_times']:
                procs.append(0)
                times.append(results['serial_times'][grid_size])

            # Add MPI times
            for nprocs in proc_counts:
                if nprocs in results['mpi_times'][grid_size]:
                    procs.append(nprocs)
                    times.append(results['mpi_times'][grid_size][nprocs])

            axes[0].plot(procs, times, 'o-', linewidth=2, markersize=8,
                        label=f'{grid_size}×{grid_size}')

    axes[0].set_xlabel('Number of MPI Processes (0 = Serial)', fontsize=12)
    axes[0].set_ylabel('Execution Time (seconds)', fontsize=12)
    axes[0].set_title('Execution Time vs Number of Processes', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks([0, 1, 2, 4, 8])
    axes[0].set_xticklabels(['Serial', '1', '2', '4', '8'])

    # Plot 1b: Speedup
    for grid_size in grid_sizes:
        if grid_size in results['speedups']:
            procs = []
            speedups = []
            for nprocs in proc_counts:
                if nprocs in results['speedups'][grid_size]:
                    procs.append(nprocs)
                    speedups.append(results['speedups'][grid_size][nprocs])

            axes[1].plot(procs, speedups, 'o-', linewidth=2, markersize=8,
                        label=f'{grid_size}×{grid_size}')

    # Add ideal speedup line
    axes[1].plot(proc_counts, proc_counts, 'k--', linewidth=2,
                label='Ideal (linear)', alpha=0.5)

    axes[1].set_xlabel('Number of MPI Processes', fontsize=12)
    axes[1].set_ylabel('Speedup (relative to serial)', fontsize=12)
    axes[1].set_title('Speedup vs Number of Processes', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks([1, 2, 4, 8])
    axes[1].set_xticklabels(['1', '2', '4', '8'])

    plt.tight_layout()
    output_file = output_dir / "benchmark_scaling.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def print_summary(results):
    """
    Print detailed summary table
    """
    print()
    print("="*70)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*70)
    print()

    for grid_size in results['grid_sizes']:
        print()
        print("-"*70)
        print(f"Grid Size: {grid_size} × {grid_size}")
        print("-"*70)

        # Serial baseline
        if grid_size in results['serial_times']:
            print(f"Serial: {results['serial_times'][grid_size]:8.3f}s")

        # MPI results
        print("-"*70)
        print(f"MPI Results:")
        print("-"*70)
        print(f"  {'Procs':>6}  {'Time (s)':>10}  {'Speedup':>10}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}")

        for nprocs in results['proc_counts']:
            if grid_size in results['mpi_times'] and nprocs in results['mpi_times'][grid_size]:
                time_val = results['mpi_times'][grid_size][nprocs]
                speedup = results['speedups'][grid_size].get(nprocs, 0)

                print(f"  {nprocs:6d}  {time_val:10.3f}  {speedup:9.2f}x")

        print()

    # Best configurations
    print("="*70)
    print("BEST CONFIGURATIONS")
    print("="*70)

    for grid_size in results['grid_sizes']:
        if grid_size in results['speedups']:
            best_speedup = 0
            best_procs = 0
            for nprocs, speedup in results['speedups'][grid_size].items():
                if speedup > best_speedup:
                    best_speedup = speedup
                    best_procs = nprocs

            if best_procs > 0:
                print(f"  {grid_size}×{grid_size}: {best_procs} processes "
                      f"(speedup: {best_speedup:.2f}x)")

    print("="*70)


def create_visualizations(results):
    """
    Create GIF and MP4 animations for all benchmark outputs

    This runs AFTER timing is complete, so it doesn't affect performance measurements.
    """
    output_dir = Path("output")
    gifs_dir = output_dir / "gifs"
    mp4s_dir = output_dir / "mp4s"

    # Create directories
    gifs_dir.mkdir(parents=True, exist_ok=True)
    mp4s_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    print("(This is NOT timed - only simulation runs are timed)")
    print()

    # Collect all npz files that were created
    npz_files = []

    # Serial files
    for grid_size in results['grid_sizes']:
        if grid_size in results['serial_times']:
            npz_file = output_dir / f"erosion_serial_{grid_size}x{grid_size}.npz"
            if npz_file.exists():
                npz_files.append(npz_file)

    # MPI files
    for grid_size in results['grid_sizes']:
        for nprocs in results['proc_counts']:
            if grid_size in results['mpi_times'] and nprocs in results['mpi_times'][grid_size]:
                npz_file = output_dir / f"erosion_mpi_{nprocs}procs_{grid_size}x{grid_size}.npz"
                if npz_file.exists():
                    npz_files.append(npz_file)

    print(f"Found {len(npz_files)} simulation outputs to visualize")
    print()

    # Create visualizations for each file
    for i, npz_file in enumerate(npz_files, 1):
        base_name = npz_file.stem
        print(f"[{i}/{len(npz_files)}] Creating visualizations for: {base_name}")

        # Create GIF
        gif_output = gifs_dir / f"{base_name}.gif"
        cmd_gif = ["python", "visualize_results.py", str(npz_file),
                   "--output-dir", str(gifs_dir), "--fps", "10"]

        # Run visualize_results.py for GIF (it creates both gif and mp4)
        result = subprocess.run(cmd_gif, capture_output=True, text=True)

        if result.returncode == 0:
            # Move the mp4 to the mp4s folder
            mp4_in_gifs = gifs_dir / f"{base_name}.mp4"
            mp4_output = mp4s_dir / f"{base_name}.mp4"

            if mp4_in_gifs.exists():
                mp4_in_gifs.rename(mp4_output)

            print(f"  ✓ Created: {gif_output.name}")
            print(f"  ✓ Created: {mp4_output.name}")
        else:
            print(f"  ✗ Failed to create visualizations")
            if result.stderr:
                print(f"    Error: {result.stderr[:200]}")

        print()

    print("="*70)
    print(f"Visualizations saved to:")
    print(f"  GIFs: {gifs_dir}/")
    print(f"  MP4s: {mp4s_dir}/")
    print("="*70)


def save_results_json(results):
    """
    Save results to JSON file for later analysis
    """
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Convert to JSON-serializable format
    json_results = {
        'grid_sizes': results['grid_sizes'],
        'proc_counts': results['proc_counts'],
        'serial_times': {str(k): v for k, v in results['serial_times'].items()},
        'mpi_times': {
            str(grid): {str(proc): time for proc, time in procs.items()}
            for grid, procs in results['mpi_times'].items()
        },
        'speedups': {
            str(grid): {str(proc): sp for proc, sp in procs.items()}
            for grid, procs in results['speedups'].items()
        }
    }

    output_file = output_dir / "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")


def main():
    print()
    print("="*70)
    print("HYDRAULIC EROSION - PERFORMANCE BENCHMARKING")
    print("="*70)
    print()
    print("This will test:")
    print("  Grid sizes: 256, 512, 1024, 2048")
    print("  MPI processes: 1, 2, 4, 8")
    print()
    print("Estimated time: 5-10 minutes")
    print()
    input("Press Enter to start benchmarking...")
    print()

    # Run benchmarks
    results = run_benchmarks()

    # Create plots
    print()
    print("="*70)
    print("Creating visualizations...")
    print("="*70)
    plot_results(results)

    # Print summary
    print_summary(results)

    # Save results
    save_results_json(results)

    # Ask about creating animations
    print()
    print("="*70)
    response = input("Create GIF/MP4 animations for all results? (y/N): ").strip().lower()

    if response == 'y':
        # Create visualizations (NOT timed - happens after all benchmarks)
        create_visualizations(results)
        print()
        print("Animations saved to:")
        print("  output/gifs/")
        print("  output/mp4s/")
    else:
        print("Skipping animation creation.")

    print()
    print("Results saved to:")
    print("  output/benchmark_results.json")
    print("  output/benchmark_scaling.png")
    print()
    print("="*70)


if __name__ == "__main__":
    main()
