from . import equilibrium
from . import boundary
from . import jtor
from . import picard

import numpy as np


def test_inoutseparatrix():
    eq = equilibrium.Equilibrium(Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0, nx=65, ny=65)

    # Two O-points, one X-point half way between them
    psi = np.exp((-((eq.R - 1.0) ** 2) - eq.Z**2) * 3) + np.exp(
        (-((eq.R - 1.0) ** 2) - (eq.Z + 1) ** 2) * 3
    )

    eq._updatePlasmaPsi(psi)

    Rin, Rout = eq.innerOuterSeparatrix()

    assert Rin >= eq.Rmin and Rout >= eq.Rmin
    assert Rin <= eq.Rmax and Rout <= eq.Rmax


def test_fixed_boundary_psi():
    # This is adapted from example 5

    eq = equilibrium.Equilibrium(
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-1.0,
        Zmax=1.0,
        nx=65,
        ny=65,
        boundary=boundary.fixedBoundary,
    )

    profiles = jtor.ConstrainPaxisIp(
        eq, 1e3, 1e5, 1.0  # Plasma pressure on axis [Pascals]  # Plasma current [Amps]
    )  # fvac = R*Bt

    # Nonlinear solve
    picard.solve(eq, profiles)

    psi = eq.psi()
    assert psi[0, 0] == 0.0  # Boundary is fixed
    assert psi[32, 32] != 0.0  # Solution is not all zero

    assert eq.psi_bndry == 0.0
    assert eq.poloidalBeta() > 0.0


def test_setSolverVcycle():
    eq = equilibrium.Equilibrium(Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0, nx=65, ny=65)

    oldsolver = eq._solver
    eq.setSolverVcycle(nlevels=2, ncycle=1, niter=5)
    assert eq._solver != oldsolver
