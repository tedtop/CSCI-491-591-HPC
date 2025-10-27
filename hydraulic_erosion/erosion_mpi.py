"""
Hydraulic Erosion with MPI
==========================

This is a step-by-step conversion of the Jupyter notebook's hydraulic erosion
model to run in parallel using MPI.

DOMAIN DECOMPOSITION STRATEGY:
We use a simple 1D decomposition - split the grid into horizontal strips,
one per MPI rank:

    Rank 0: [========]
    Rank 1: [========]
    Rank 2: [========]
    Rank 3: [========]

Each rank needs the row above and below (ghost/halo cells) from neighbors.
"""

from mpi4py import MPI
import numpy as np
import sys
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
# STEP 1: Domain Decomposition
# ============================================================================

def setup_domain_decomposition(nx_global, ny_global, comm):
    """
    Split the global grid among MPI processes.

    Strategy: 1D decomposition in y-direction
    - Each rank gets a horizontal strip
    - Strips are roughly equal size
    - Last rank gets any remainder rows

    Returns:
    --------
    domain_info : dict with keys
        - ny_local: number of rows this rank owns
        - start_row: first row index in global grid
        - end_row: last row index (exclusive) in global grid
        - nx_local: number of columns (same for all ranks)
        - top_neighbor: rank above us (or MPI.PROC_NULL if none)
        - bottom_neighbor: rank below us (or MPI.PROC_NULL if none)
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Divide rows among ranks
    base_rows = ny_global // size
    extra_rows = ny_global % size

    # Figure out our slice of rows
    if rank < extra_rows:
        # First 'extra_rows' ranks get one extra row
        ny_local = base_rows + 1
        start_row = rank * ny_local
    else:
        ny_local = base_rows
        start_row = rank * base_rows + extra_rows

    end_row = start_row + ny_local

    # Who are our neighbors?
    top_neighbor = rank - 1 if rank > 0 else MPI.PROC_NULL
    bottom_neighbor = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    domain_info = {
        'ny_local': ny_local,
        'start_row': start_row,
        'end_row': end_row,
        'nx_local': nx_global,  # x-direction not decomposed
        'top_neighbor': top_neighbor,
        'bottom_neighbor': bottom_neighbor,
        'rank': rank,
        'size': size
    }

    return domain_info


def extract_local_domain(global_array, domain_info):
    """
    Extract this rank's portion from a global array.

    Adds ghost/halo rows:
    - Row 0: ghost (will receive from top neighbor)
    - Rows 1 to ny_local: actual data for this rank
    - Row ny_local+1: ghost (will receive from bottom neighbor)

    So local array shape is (ny_local + 2, nx_local)
    """
    start = domain_info['start_row']
    end = domain_info['end_row']

    # Get our rows
    local_data = global_array[start:end, :].copy()

    # Add ghost rows (pad with edge values as initial guess)
    local_with_ghosts = np.pad(local_data, ((1, 1), (0, 0)), mode='edge')

    return local_with_ghosts


# ============================================================================
# STEP 2: Halo Exchange
# ============================================================================

def exchange_halo(field, domain_info, comm):
    """
    Exchange boundary data with neighboring ranks.

    This is the CRITICAL MPI operation for this simulation!

    What happens:
    1. Send my bottom-most real row to the rank below me
    2. Receive their top-most real row into my bottom ghost
    3. Send my top-most real row to the rank above me
    4. Receive their bottom-most real row into my top ghost

    We use Sendrecv to do this efficiently (send and receive in one call).

    Field layout:
        field[0, :]     <- top ghost (receive from top_neighbor)
        field[1, :]     <- my first real row (send to top_neighbor)
        ...
        field[-2, :]    <- my last real row (send to bottom_neighbor)
        field[-1, :]    <- bottom ghost (receive from bottom_neighbor)
    """
    top_neighbor = domain_info['top_neighbor']
    bottom_neighbor = domain_info['bottom_neighbor']

    # Create contiguous buffers (needed for MPI)
    send_to_top = field[1, :].copy()       # My first row
    send_to_bottom = field[-2, :].copy()   # My last row

    # Allocate receive buffers
    recv_from_top = np.empty_like(field[0, :])
    recv_from_bottom = np.empty_like(field[-1, :])

    # Exchange with top neighbor
    comm.Sendrecv(
        sendbuf=send_to_top,
        dest=top_neighbor,
        recvbuf=recv_from_top,
        source=top_neighbor
    )

    # Exchange with bottom neighbor
    comm.Sendrecv(
        sendbuf=send_to_bottom,
        dest=bottom_neighbor,
        recvbuf=recv_from_bottom,
        source=bottom_neighbor
    )

    # Update ghost cells
    field[0, :] = recv_from_top
    field[-1, :] = recv_from_bottom


# ============================================================================
# STEP 3: The Flow Update (from notebook, minimally modified)
# ============================================================================

def update_flow_mpi(z, h, r, u, v, fL, fR, fT, fB, dt, k_e, domain_info, comm):
    """
    One time step of the hydraulic erosion model - MPI version.

    z,   # Terrain height [ny, nx]
    h,   # Water height [ny, nx]
    r,   # Rainfall rate [ny, nx]
    u,   # x-velocity [ny, nx]
    v,   # y-velocity [ny, nx]
    fL,  # Flux to left   neighbor [ny, nx]
    fR,  # Flux to right  neighbor [ny, nx]
    fT,  # Flux to top    neighbor [ny, nx]
    fB,  # Flux to bottom neighbor [ny, nx]
    dt,  # Time step (s)
    k_e = K_E  # Evaporation rate (height units per second, constant)

    Almost identical to the serial version, except:
    - We first call exchange_halo() to get neighbor data
    - We apply boundary conditions only on physical domain edges (not betwen ranks)

    All arrays have shape (ny_local + 2, nx_local) with ghost cells.
    """

    # --- EXCHANGE HALOS ---
    # We need neighbor data for computing gradients
    exchange_halo(h, domain_info, comm)
    exchange_halo(z, domain_info, comm)

    # --- REST IS SAME AS NOTEBOOK ---
    ny_local_with_ghosts, nx_local = z.shape

    # Interior cells (excluding ghosts)
    inner = (slice(1, -1), slice(1, -1))

    # Pre-rain surface
    H = z + h

    # Rainfall
    h += r * dt

    # Flux growth
    flux_factor = dt * A_PIPE / L_PIPE * G

    # Height differences
    dhL = H[inner] - H[1:-1, 0:-2]   # Left neighbor
    dhR = H[inner] - H[1:-1, 2:]     # Right neighbor
    dhT = H[inner] - H[0:-2, 1:-1]   # Top neighbor
    dhB = H[inner] - H[2:, 1:-1]     # Bottom neighbor

    # Update fluxes
    fL[inner] = np.maximum(0.0, fL[inner] + dhL * flux_factor)
    fR[inner] = np.maximum(0.0, fR[inner] + dhR * flux_factor)
    fT[inner] = np.maximum(0.0, fT[inner] + dhT * flux_factor)
    fB[inner] = np.maximum(0.0, fB[inner] + dhB * flux_factor)

    # Flux adjustment to prevent negative water
    sum_f_inner = fL[inner] + fR[inner] + fT[inner] + fB[inner]
    adj = np.ones_like(sum_f_inner)
    mask = sum_f_inner > 0.0
    adj[mask] = h[inner][mask] * LX * LY / (sum_f_inner[mask] * dt)
    adj = np.minimum(1.0, adj)

    fL[inner] *= adj
    fR[inner] *= adj
    fT[inner] *= adj
    fB[inner] *= adj

    # --- BOUNDARY CONDITIONS ---
    # Zero boundary fluxes at PHYSICAL domain edges only
    # Left and right column edges - physical for all ranks
    fL[:, 0] = 0.0
    fR[:, -1] = 0.0

    # Top row edge - only for topmost rank
    if domain_info['top_neighbor'] == MPI.PROC_NULL:
        fT[1, :] = 0.0  # Row 1 is first real row (0 is ghost)

    # Bottom row edge - only for bottommost rank
    if domain_info['bottom_neighbor'] == MPI.PROC_NULL:
        fB[-2, :] = 0.0  # Row -2 is last real row (-1 is ghost)

    # --- CONTINUITY EQUATION ---
    sum_f_in = (
        fR[1:-1, 0:-2] +   # from left
        fT[2:, 1:-1] +     # from bottom
        fL[1:-1, 2:] +     # from right
        fB[0:-2, 1:-1]     # from top
    )
    sum_f_out = fL[inner] + fR[inner] + fT[inner] + fB[inner]

    dvol = dt * (sum_f_in - sum_f_out)
    dh_inner = dvol / (LX * LY)
    h2_inner = h[inner] + dh_inner

    # --- VELOCITIES ---
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

    # --- EVAPORATION ---
    h[inner] = np.maximum(0.0, h2_inner - k_e * dt)

    return z, h, r, u, v, fL, fR, fT, fB, dt, k_e


# ============================================================================
# STEP 4: Gathering Results
# ============================================================================

def gather_global_field(local_field, domain_info, comm):
    """
    Reconstruct the global field from all ranks' local pieces.

    Returns:
    --------
    global_field : np.ndarray or None
        Full global array on rank 0, None on other ranks
    """
    # Strip ghost cells to get only real data
    local_real = local_field[1:-1, :]

    # Gather all pieces to rank 0
    all_pieces = comm.gather(local_real, root=0)

    if domain_info['rank'] == 0:
        # Stack vertically to reconstruct
        return np.vstack(all_pieces)
    else:
        return None


# ============================================================================
# STEP 5: Main Simulation Loop
# ============================================================================

def simulate_mpi(z_global, T, dt, rainfall_rate, domain_info, comm):
    """
    Run the full MPI simulation.

    Parameters:
    -----------
    z_global : np.ndarray
        Full terrain (must be same on all ranks)
    T : float
        Total simulation time
    dt : float
        Time step size
    rainfall_rate : float
        Rain rate
    domain_info : dict
        From setup_domain_decomposition
    comm : MPI.Comm
        MPI communicator

    Returns:
    --------
    times : list of float (rank 0 only)
    H_hist : list of np.ndarray (rank 0 only)
        Water height snapshots
    """
    rank = domain_info['rank']

    # Extract local domain
    z_local = extract_local_domain(z_global, domain_info)
    ny_local_ghosts, nx_local = z_local.shape

    # Initialize state
    h_local = np.zeros_like(z_local)
    r_local = np.full_like(z_local, rainfall_rate)
    u_local = np.zeros_like(z_local)
    v_local = np.zeros_like(z_local)
    fL_local = np.zeros_like(z_local)
    fR_local = np.zeros_like(z_local)
    fT_local = np.zeros_like(z_local)
    fB_local = np.zeros_like(z_local)

    # Time stepping
    n_steps = int(T / dt)
    save_interval = max(1, n_steps // 50)  # Save ~50 snapshots (same as serial)
    times = []
    H_hist = []

    if rank == 0:
        print(f"Starting MPI simulation:")
        print(f"  Grid: {z_global.shape}")
        print(f"  MPI ranks: {domain_info['size']}")
        print(f"  Steps: {n_steps}")
        print(f"  Time: 0 to {T}")

    # Barrier to sync before timing
    comm.Barrier()
    t_start = MPI.Wtime()

    for step in range(n_steps):
        t = step * dt

        # Rainfall schedule (same as notebook)
        if t < T / 2.0:
            r_local.fill(rainfall_rate)
        else:
            r_local.fill(0.0)

        # One time step
        update_flow_mpi(
            z_local, h_local, r_local, u_local, v_local,
            fL_local, fR_local, fT_local, fB_local,
            dt, K_E, domain_info, comm
        )

        # Save snapshots periodically (same interval as serial)
        if step % save_interval == 0 or step == n_steps - 1:
            h_global = gather_global_field(h_local, domain_info, comm)
            if rank == 0:
                times.append(t)
                H_hist.append(h_global.copy())

    comm.Barrier()
    t_end = MPI.Wtime()

    if rank == 0:
        print(f"Simulation complete in {t_end - t_start:.2f} seconds")

    return times, H_hist


# ============================================================================
# STEP 6: Testing
# ============================================================================

def test_simple_grid():
    """
    Test with a simple ramp + noise terrain (from notebook)
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parse command-line arguments
    if len(sys.argv) >= 2:
        nx = ny = int(sys.argv[1])
    else:
        nx = ny = 256  # Default

    # Parameters
    T = 50.0
    dt = 0.2
    rainfall_rate = 1e-2

    # Create output directory
    if rank == 0:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

    # Setup decomposition
    domain_info = setup_domain_decomposition(nx, ny, comm)

    if rank == 0:
        # Create terrain (simple version for testing)
        np.random.seed(42)
        z = np.random.rand(ny, nx) * 0.5
        # Add a slope
        z += np.linspace(0., 2., ny)[:, None]
    else:
        z = None

    # Broadcast terrain to all ranks
    z = comm.bcast(z, root=0)

    # Run simulation
    times, H_hist = simulate_mpi(z, T, dt, rainfall_rate, domain_info, comm)

    # Save results (rank 0 only)
    if rank == 0:
        output_file = Path("output") / f"erosion_mpi_{size}procs_{ny}x{nx}.npz"
        print(f"Saved {len(H_hist)} snapshots")
        np.savez(output_file,
                 times=times,
                 H_hist=np.array(H_hist),
                 z=z)
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    test_simple_grid()
