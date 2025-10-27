"""
Visualize results from MPI simulation

This script loads the .npz output file and creates visualizations
similar to the notebook animations.

Usage:
    python visualize_results.py erosion_mpi_output.npz
    python visualize_results.py erosion_mpi_output.npz --output-dir results/
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
import sys
import os
from pathlib import Path

def load_results(filename):
    """Load simulation results from npz file"""
    data = np.load(filename)
    return data['times'], data['H_hist'], data['z']


def create_static_plots(times, H_hist, z):
    """Create static plots showing initial, middle, and final states"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Initial state
    axes[0, 0].imshow(z, cmap='terrain')
    axes[0, 0].set_title('Terrain Height (z)')
    axes[0, 0].colorbar = plt.colorbar(axes[0, 0].images[0], ax=axes[0, 0])

    # Initial water
    axes[0, 1].imshow(H_hist[0], cmap='Blues')
    axes[0, 1].set_title(f'Water at t={times[0]:.1f}s')
    plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1])

    # Middle water
    mid_idx = len(H_hist) // 2
    axes[0, 2].imshow(H_hist[mid_idx], cmap='Blues')
    axes[0, 2].set_title(f'Water at t={times[mid_idx]:.1f}s')
    plt.colorbar(axes[0, 2].images[0], ax=axes[0, 2])

    # Final water
    axes[1, 0].imshow(H_hist[-1], cmap='Blues')
    axes[1, 0].set_title(f'Water at t={times[-1]:.1f}s')
    plt.colorbar(axes[1, 0].images[0], ax=axes[1, 0])

    # Water evolution at center point
    ny, nx = H_hist[0].shape
    center_h = H_hist[:, ny//2, nx//2]
    axes[1, 1].plot(times, center_h, 'b-', linewidth=2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Water Height')
    axes[1, 1].set_title('Water Height at Center')
    axes[1, 1].grid(True)

    # Total water over time
    total_water = np.sum(H_hist, axis=(1, 2))
    axes[1, 2].plot(times, total_water, 'r-', linewidth=2)
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('Total Water Volume')
    axes[1, 2].set_title('Total Water Volume Over Time')
    axes[1, 2].grid(True)

    plt.tight_layout()
    return fig


def create_animation(times, H_hist, output_file='water_animation.mp4', fps=10):
    """
    Create an animation of the water height evolution

    Parameters
    ----------
    times : array
        Time values
    H_hist : array (n_times, ny, nx)
        Water height at each time
    output_file : str
        Output filename (.mp4 or .gif)
    fps : int
        Frames per second
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Set color scale based on data range
    vmin = 0
    vmax = np.percentile(H_hist, 99)  # Use 99th percentile to avoid outliers

    # Initial frame
    im = ax.imshow(H_hist[0], cmap='Blues', vmin=vmin, vmax=vmax, animated=True)
    plt.colorbar(im, ax=ax, label='Water Height')
    title = ax.set_title(f'Hydraulic Erosion\nt = {times[0]:.2f} s')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    def update(frame):
        """Update function for animation"""
        im.set_array(H_hist[frame])
        title.set_text(f'Hydraulic Erosion\nt = {times[frame]:.2f} s (frame {frame}/{len(times)-1})')
        return [im, title]

    print(f"Creating animation with {len(times)} frames...")
    anim = FuncAnimation(fig, update, frames=len(times),
                        interval=1000//fps, blit=True)

    # Save animation
    if output_file.endswith('.gif'):
        print(f"Saving as GIF: {output_file}")
        writer = PillowWriter(fps=fps)
        anim.save(output_file, writer=writer)
    else:
        print(f"Saving as MP4: {output_file}")
        anim.save(output_file, writer='ffmpeg', fps=fps)

    print(f"Animation saved to: {output_file}")
    plt.close(fig)
    return anim


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_results.py <npz_file> [--output-dir <directory>] [--fps <fps>]")
        print("\nExamples:")
        print("  python visualize_results.py erosion_mpi_output.npz")
        print("  python visualize_results.py erosion_mpi_output.npz --output-dir output/")
        print("  python visualize_results.py erosion_mpi_output.npz --fps 20")
        sys.exit(1)

    npz_file = sys.argv[1]

    # Parse optional arguments
    output_dir = "output"
    fps = 10

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--output-dir' and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--fps' and i + 1 < len(sys.argv):
            fps = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get base name for output files
    base_name = Path(npz_file).stem  # Gets filename without extension

    print("="*70)
    print("HYDRAULIC EROSION VISUALIZATION")
    print("="*70)
    print(f"Input file: {npz_file}")
    print(f"Output directory: {output_dir}/")
    print()

    print(f"Loading results from: {npz_file}")
    times, H_hist, z = load_results(npz_file)

    print(f"\nSimulation info:")
    print(f"  Grid size: {z.shape}")
    print(f"  Time steps: {len(times)}")
    print(f"  Time range: {times[0]:.2f} to {times[-1]:.2f} seconds")
    print(f"  Final total water: {H_hist[-1].sum():.2f}")
    print(f"  Final max water height: {H_hist[-1].max():.6f}")
    print(f"  Final mean water height: {H_hist[-1].mean():.6f}")
    print()

    # Create animations (both MP4 and GIF)
    print("Creating animations...")
    print(f"  FPS: {fps}")
    print()

    # MP4 animation
    mp4_file = os.path.join(output_dir, f'{base_name}.mp4')
    print(f"[1/2] Creating MP4 animation...")
    create_animation(times, H_hist, output_file=mp4_file, fps=fps)

    # GIF animation
    gif_file = os.path.join(output_dir, f'{base_name}.gif')
    print(f"[2/2] Creating GIF animation...")
    create_animation(times, H_hist, output_file=gif_file, fps=fps)

    print()
    print("="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"Outputs saved to: {output_dir}/")
    print(f"  - {base_name}.mp4")
    print(f"  - {base_name}.gif")
    print()
    print("View with:")
    print(f"  open {mp4_file}")
    print(f"  open {gif_file}")
    print("="*70)


if __name__ == "__main__":
    main()
