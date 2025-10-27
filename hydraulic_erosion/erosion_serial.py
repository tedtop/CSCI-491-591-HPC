"""
Hydraulic Erosion - Serial NumPy Version
=========================================

This is the EXACT implementation from the Jupyter notebook.
It serves as the reference/ground truth for comparing against the MPI version.

Usage:
    python erosion_serial.py [grid_size] [output_file]

Examples:
    python erosion_serial.py 256 serial_output.npz
    python erosion_serial.py 512
"""

import numpy as np
import sys
import time
from pathlib import Path

# ============================================================================
# CONSTANTS (from the notebook)
# ============================================================================
A_PIPE = 0.6   # virtual pipe cross section
G = 9.81       # gravitational acceleration
L_PIPE = 1     # virtual pipe length
LX = 1         # horizontal distance between grid points
LY = 1         # vertical distance between grid points
K_E = 0.003    # evaporation constant


# ============================================================================
# FLOW UPDATE (from notebook - NumPy version)
# ============================================================================

def update_flow_numpy(z, h, r, u, v, fL, fR, fT, fB, dt, k_e=K_E):
    """
    One Euler step of the 2D shallow "virtual pipes" water-flow model (no erosion),
    updating all fields **in place** (NumPy vectorized version).

    This is EXACTLY the code from the notebook cell "update_flow_numpy".

    Parameters
    ----------
    z, h, r, u, v, fL, fR, fT, fB : np.ndarray, shape (ny, nx)
        State arrays, all updated in place. `z` is unchanged here.
    dt : float
        Time step (seconds).
    k_e : float, optional
        Constant evaporation rate (height units per second).

    Returns
    -------
    z, h, r, u, v, fL, fR, fT, fB, dt, k_e
        Same objects passed in (all but `z` modified as described).
    """
    ny, nx = z.shape
    inner = (slice(1, ny - 1), slice(1, nx - 1))

    # Pre-rain surface height
    H = z + h

    # Rainfall
    h += r * dt

    # Outflow flux growth
    flux_factor = dt * A_PIPE / L_PIPE * G

    dhL = H[inner] - H[1:-1, 0:-2]
    dhR = H[inner] - H[1:-1, 2:]
    dhT = H[inner] - H[0:-2, 1:-1]
    dhB = H[inner] - H[2:, 1:-1]

    fL[inner] = np.maximum(0.0, fL[inner] + dhL * flux_factor)
    fR[inner] = np.maximum(0.0, fR[inner] + dhR * flux_factor)
    fT[inner] = np.maximum(0.0, fT[inner] + dhT * flux_factor)
    fB[inner] = np.maximum(0.0, fB[inner] + dhB * flux_factor)

    # Adjustment
    sum_f_inner = fL[inner] + fR[inner] + fT[inner] + fB[inner]
    adj = np.ones_like(sum_f_inner)
    mask = sum_f_inner > 0.0
    adj[mask] = h[inner][mask] * LX * LY / (sum_f_inner[mask] * dt)
    adj = np.minimum(1.0, adj)
    fL[inner] *= adj
    fR[inner] *= adj
    fT[inner] *= adj
    fB[inner] *= adj

    # Zero boundary fluxes
    fL[0, :] = 0.0
    fR[-1, :] = 0.0
    fT[:, 0] = 0.0
    fB[:, -1] = 0.0

    # Continuity
    sum_f_in = (
        fR[1:-1, 0:-2] +
        fT[2:, 1:-1] +
        fL[1:-1, 2:] +
        fB[0:-2, 1:-1]
    )
    sum_f_out = fL[inner] + fR[inner] + fT[inner] + fB[inner]
    dvol = dt * (sum_f_in - sum_f_out)
    dh_inner = dvol / (LX * LY)
    h2_inner = h[inner] + dh_inner

    # Velocities
    h_mean = h[inner] + 0.5 * dh_inner
    dwx = (
        fR[1:-1, 0:-2] - fL[1:-1, 1:-1] +
        fR[1:-1, 1:-1] - fL[1:-1, 2:]
    )
    dwy = (
        fB[0:-2, 1:-1] - fT[1:-1, 1:-1] +
        fB[1:-1, 1:-1] - fT[2:, 1:-1]
    )

    u.fill(0.0)
    v.fill(0.0)
    pos = h_mean > 0.0
    u_i = np.zeros_like(h_mean)
    v_i = np.zeros_like(h_mean)
    u_i[pos] = dwx[pos] / (LY * h_mean[pos])
    v_i[pos] = dwy[pos] / (LX * h_mean[pos])
    u[inner] = u_i
    v[inner] = v_i

    # Evaporation
    h[inner] = np.maximum(0.0, h2_inner - k_e * dt)

    return z, h, r, u, v, fL, fR, fT, fB, dt, k_e


# ============================================================================
# SIMULATION LOOP
# ============================================================================

def simulate_serial(z, T, dt, rainfall_rate, verbose=True):
    """
    Run serial hydraulic erosion simulation

    Parameters
    ----------
    z : np.ndarray, shape (ny, nx)
        Initial terrain height
    T : float
        Total simulation time
    dt : float
        Time step
    rainfall_rate : float
        Constant rainfall rate
    verbose : bool
        Print progress

    Returns
    -------
    times : list
        Time values at saved snapshots
    H_history : list of np.ndarray
        Water height snapshots
    """
    ny, nx = z.shape

    # Initialize state
    h = np.zeros_like(z)
    r = np.full_like(z, rainfall_rate)
    u = np.zeros_like(z)
    v = np.zeros_like(z)
    fL = np.zeros_like(z)
    fR = np.zeros_like(z)
    fT = np.zeros_like(z)
    fB = np.zeros_like(z)

    # Time-stepping
    n_steps = int(T / dt)
    save_interval = max(1, n_steps // 50)  # Save ~50 snapshots

    times = []
    H_history = []

    if verbose:
        print(f"\n=== Starting Serial Simulation ===")
        print(f"Grid: {ny} × {nx}")
        print(f"Time steps: {n_steps}")
        print(f"Simulation time: {T} s")
        print(f"Time step: {dt} s")
        print()

    # Start timer
    t_start = time.time()

    for step in range(n_steps):
        t = step * dt

        # Rainfall schedule (same as notebook/MPI version)
        if t < T / 2.0:
            r.fill(rainfall_rate)
        else:
            r.fill(0.0)

        # Update flow
        update_flow_numpy(z, h, r, u, v, fL, fR, fT, fB, dt, K_E)

        # Save snapshots
        if step % save_interval == 0 or step == n_steps - 1:
            times.append(t)
            H_history.append(h.copy())
            if verbose and step % (save_interval * 5) == 0:
                print(f"  Step {step}/{n_steps}, t={t:.2f}s")

    # Stop timer
    t_end = time.time()
    elapsed = t_end - t_start

    if verbose:
        print()
        print(f"=== Simulation Complete ===")
        print(f"Total time: {elapsed:.3f} s")
        print(f"Snapshots saved: {len(H_history)}")
        print()

    return times, H_history


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main function for testing serial implementation
    """
    # Parse command-line arguments
    if len(sys.argv) >= 2:
        grid_size = int(sys.argv[1])
    else:
        grid_size = 256  # Default

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Generate output filename with parameters
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        output_file = output_dir / f"erosion_serial_{grid_size}x{grid_size}.npz"

    print("="*60)
    print("Hydraulic Erosion - Serial Version")
    print("="*60)
    print(f"Grid size: {grid_size} × {grid_size}")
    print(f"Output file: {output_file}")

    # Create initial terrain (simple version for testing)
    np.random.seed(42)  # Ensure reproducibility
    z = np.random.rand(grid_size, grid_size) * 0.5
    # Add a slope so water flows
    z += np.linspace(0., 2., grid_size)[:, None]

    print(f"\nTerrain stats:")
    print(f"  Min height: {z.min():.3f}")
    print(f"  Max height: {z.max():.3f}")
    print(f"  Mean height: {z.mean():.3f}")

    # Run simulation
    times, H_history = simulate_serial(
        z,
        T=50.0,
        dt=0.2,
        rainfall_rate=0.01,
        verbose=True
    )

    # Save results
    np.savez(output_file,
             times=np.array(times),
             H_hist=np.array(H_history),
             z=z)

    print(f"Results saved to: {output_file}")
    print(f"\nFinal water stats:")
    print(f"  Total water: {H_history[-1].sum():.2f}")
    print(f"  Max height: {H_history[-1].max():.6f}")
    print(f"  Mean height: {H_history[-1].mean():.6f}")
    print("="*60)


if __name__ == "__main__":
    main()
