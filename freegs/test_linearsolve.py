"""
Tests of the linear solver
"""

import numpy as np

from . import multigrid

# Laplacian in 2D


def test_direct_laplacian():
    nx = 65
    ny = 65

    Lx = 2.0
    Ly = 3.0

    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    xx, yy = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))

    solution = np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.4**2)

    A = multigrid.LaplacianOp()
    rhs = A(solution, dx, dy)

    # Copy boundaries
    rhs[:, 0] = solution[:, 0]
    rhs[0, :] = solution[0, :]
    rhs[:, -1] = solution[:, -1]
    rhs[-1, :] = solution[-1, :]

    solve = multigrid.MGDirect(multigrid.LaplaceSparse(Lx, Ly)(nx, ny))
    x = solve(None, rhs)

    assert np.allclose(x, solution)


def test_multigrid_laplacian():
    nx = 65
    ny = 65

    Lx = 2.0
    Ly = 3.0

    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    xx, yy = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))

    solution = np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.4**2)

    A = multigrid.LaplacianOp()
    rhs = A(solution, dx, dy)

    # Copy boundaries
    rhs[:, 0] = solution[:, 0]
    rhs[0, :] = solution[0, :]
    rhs[:, -1] = solution[:, -1]
    rhs[-1, :] = solution[-1, :]

    solve = multigrid.createVcycle(
        nx,
        ny,
        multigrid.LaplaceSparse(Lx, Ly),
        ncycle=50,
        niter=50,
        nlevels=4,
        direct=True,
    )

    xinput = np.zeros(rhs.shape)
    xinput[:, 0] = solution[:, 0]
    xinput[0, :] = solution[0, :]
    xinput[:, -1] = solution[:, -1]
    xinput[-1, :] = solution[-1, :]
    x = solve(xinput, rhs)

    assert np.allclose(x, solution, atol=1e-6)
