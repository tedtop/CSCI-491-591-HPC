from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

def init_field(nx, ny, hot=1.0, block=10):
    """
    Initialize a 2D field with a hot square in the center.

    Parameters
    ----------
    nx : int
        Number of columns.
    ny : int
        Number of rows.
    hot : float, optional
        Value to assign inside the hot block (default 1.0).
    block : int, optional
        Approximate size of the hot block edge (default 10).

    Returns
    -------
    u : ndarray of shape (ny, nx)
        Array with the hot block set to `hot`, rest zero.
    """
    u = np.zeros((ny, nx), dtype=np.float64)
    cx, cy = nx // 2, ny // 2
    b = block // 2
    u[cy-b:cy+b, cx-b:cx+b] = hot
    return u

def halo_exchange(comm, u):
    """
    Exchange top/bottom halo rows with neighboring ranks.

    Parameters
    ----------
    comm : MPI.Comm
        MPI communicator.
    u : ndarray of shape (my+2, nx)
        Local array with one ghost row at top and bottom.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    up   = rank - 1 if rank > 0 else MPI.PROC_NULL
    down = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    comm.Sendrecv(u[1, :],  dest=up,   sendtag=0,
                  recvbuf=u[-1, :], source=down, recvtag=0)
    comm.Sendrecv(u[-2, :], dest=down, sendtag=1,
                  recvbuf=u[0, :],    source=up,   recvtag=1)

def run_sim(comm, u0=None, steps=500, alpha=0.24):
    """
    Distribute the initial field, run Jacobi iterations, and gather results.

    Parameters
    ----------
    comm : MPI.Comm
        MPI communicator.
    u0 : ndarray or None
        Full initial condition on root; None on other ranks.
    steps : int, optional
        Number of Jacobi iterations (default 500).
    alpha : float, optional
        Diffusion coefficient for the update (default 0.24).

    Returns
    -------
    U_final : ndarray or None
        Global field on root, None on other ranks.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        ny, nx = map(int, u0.shape)
    else:
        ny, nx = None, None
    ny, nx = comm.bcast((ny, nx), root=0)

    edges = np.floor(np.linspace(0, ny, size + 1)).astype(np.int64)
    starts = edges[:-1]
    stops  = edges[1:]
    y0, y1 = int(starts[rank]), int(stops[rank])
    my = y1 - y0

    if rank == 0:
        counts = (stops - starts) * nx
        displs = starts * nx
        sendbuf = [u0, counts, displs, MPI.DOUBLE]
    else:
        sendbuf = None

    local = np.empty((my, nx), dtype=np.float64)
    comm.Scatterv(sendbuf, local, root=0)

    u = np.zeros((my + 2, nx), dtype=np.float64)
    u[1:-1, :] = local

    for _ in range(steps):
        halo_exchange(comm, u)
        un = u.copy()
        center = u[1:-1, 1:-1]
        left   = u[1:-1, :-2]
        right  = u[1:-1, 2:]
        up     = u[0:-2, 1:-1]
        down   = u[2:, 1:-1]
        un[1:-1, 1:-1] = center + alpha*(left + right + up + down - 4.0*center)
        u = un

    interior = u[1:-1, :]

    if rank == 0:
        recvbuf = np.empty((ny, nx), dtype=u.dtype)
        mpi_dt = MPI._typedict[interior.dtype.char]
        comm.Gatherv(interior, [recvbuf, counts, displs, mpi_dt], root=0)
        U_final = recvbuf
    else:
        comm.Gatherv(interior, None, root=0)
        U_final = None

    return U_final


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    nx, ny = 1024, 1024
    if rank == 0:
        u0 = init_field(nx=nx, ny=ny, hot=1.0, block=250)
    else:
        u0 = None

    u1 = run_sim(comm, u0=u0, steps=5000, alpha=0.24)

    if rank == 0:
        plt.imshow(u1)
        plt.colorbar()
        plt.show()

