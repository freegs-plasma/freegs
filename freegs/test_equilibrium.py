from . import equilibrium
from . import boundary
from . import jtor
from . import picard
from . import control
from .machine import TestTokamak

import numpy as np

def test_inoutseparatrix():
    eq = equilibrium.Equilibrium(Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0, nx=65, ny=65)

    # Two O-points, one X-point half way between them
    psi = np.exp((-((eq.R - 1.0) ** 2) - eq.Z**2) * 3) + np.exp(
        (-((eq.R - 1.0) ** 2) - (eq.Z + 1) ** 2) * 3
    )

    eq._updatePlasmaPsi(psi)

    Rin, Rout = eq.innerOuterSeparatrix()

    assert Rin >= eq.Rmin
    assert Rout >= eq.Rmin
    assert Rin <= eq.Rmax
    assert Rout <= eq.Rmax


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

    # Plasma pressure on axis [Pascals]  # Plasma current [Amps]
    profiles = jtor.ConstrainPaxisIp(eq, 1e3, 1e5, 1.0)  # fvac = R*Bt

    # Nonlinear solve
    picard.solve(eq, profiles)

    psi = eq.psi()
    assert psi[0, 0] == 0.0  # Boundary is fixed
    assert psi[32, 32] != 0.0  # Solution is not all zero

    assert eq.psi_bndry == 0.0
    assert eq.poloidalBeta() > 0.0


def test_fixed_boundary_psi_check_limited():
    # This is adapted from example 5

    tokamak = TestTokamak()

    eq = equilibrium.Equilibrium(
        tokamak=tokamak,
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-1.0,
        Zmax=1.0,
        nx=65,
        ny=65,
        boundary=boundary.fixedBoundary,
        check_limited=True,
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


def _solve_diverted(nx, ny):
    # Adapted from example 1: free boundary, lower single null, diverted
    # (the separatrix does not touch the TestTokamak wall)
    tokamak = TestTokamak()
    eq = equilibrium.Equilibrium(
        tokamak=tokamak, Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0, nx=nx, ny=ny
    )
    profiles = jtor.ConstrainPaxisIp(eq, 1e3, 2e5, 2.0)
    constrain = control.constrain(
        xpoints=[(1.1, -0.6), (1.1, 0.8)], isoflux=[(1.1, -0.6, 1.1, 0.6)]
    )
    picard.solve(eq, profiles, constrain, check_limited=True)
    return eq


def test_check_limited_diverted_plasma():
    eq = _solve_diverted(65, 65)

    # The limiter flux must be the flux at the reported contact point
    assert np.isclose(eq.psi_limit, eq.psiRZ(eq.Rlim, eq.Zlim))

    # Every wall point is outside the separatrix, so the plasma is diverted
    psi_wall = eq.psiRZ(eq.tokamak.limit_points_R, eq.tokamak.limit_points_Z)
    assert np.all(psi_wall < eq.psi_xpt)
    assert not eq.is_limited
    assert eq.psi_bndry == eq.psi_xpt


def test_check_limited_non_square_grid():
    # psi[nx, ny] must be passed to the limiter interpolation untransposed;
    # with nx != ny a transposed array raises in RectBivariateSpline
    eq = _solve_diverted(65, 33)
    assert not eq.is_limited


def test_refine_and_coarsen_re_solve():
    eq = _solve_diverted(33, 65)

    eq_fine = equilibrium.refine(eq)
    assert eq_fine.R.shape == (65, 129)
    assert eq_fine._profiles.eq is eq_fine
    assert eq._profiles.eq is eq
    picard.solve(eq_fine, eq_fine._profiles, eq_fine.control)

    eq_coarse = equilibrium.coarsen(eq)
    assert eq_coarse.R.shape == (17, 33)
    assert eq_coarse._profiles.eq is eq_coarse
    picard.solve(eq_coarse, eq_coarse._profiles, eq_coarse.control)


def test_new_domain_non_square_grid():
    eq = _solve_diverted(33, 65)
    eq_new = equilibrium.newDomain(eq, Rmin=0.2, Rmax=2.1)
    assert eq_new.R.shape == (33, 65)
    assert eq_new.Rmin == 0.2
    assert eq_new.Rmax == 2.1
    assert eq_new._profiles.eq is eq_new
