from itertools import pairwise

import numpy as np
from numpy.polynomial.legendre import leggauss

import freegs
from freegs.boundary import freeBoundaryHagenow
from freegs.gradshafranov import Greens

# Compact smooth current blob of radius A centred at (R0, 0) inside the box
R0, A = 1.2, 0.5


def blob_current(R, Z):
    r2 = ((R - R0) ** 2 + Z**2) / A**2
    inside = r2 < 1.0
    Jtor = np.zeros_like(R)
    Jtor[inside] = 1e6 * np.exp(1.0 - 1.0 / (1.0 - r2[inside]))
    return Jtor


def boundary_psi_reference(R, Z):
    """Direct Green's function integral of the blob current at (R, Z) points
    outside its support: smooth integrand, polar Gauss-Legendre quadrature"""
    xr, wr = leggauss(60)
    xt, wt = leggauss(100)
    rr, tt = np.meshgrid(0.5 * A * (xr + 1.0), np.pi * (xt + 1.0), indexing="ij")
    weight = np.outer(0.5 * A * wr, np.pi * wt) * rr
    Rq = R0 + rr * np.cos(tt)
    Zq = rr * np.sin(tt)
    Jq = blob_current(Rq, Zq)
    return np.array([np.sum(weight * Jq * Greens(Rq, Zq, r, z)) for r, z in zip(R, Z)])


def test_hagenow_boundary_converges():
    """Boundary psi from von Hagenow's method converges to the direct
    Green's function integral of the plasma current, at (at least)
    the order of the 4th-order interior solver"""

    errors = []
    for nx in [33, 65, 129, 257]:
        eq = freegs.Equilibrium(
            tokamak=freegs.machine.TestTokamak(),
            Rmin=0.2,
            Rmax=2.2,
            Zmin=-1.0,
            Zmax=1.0,
            nx=nx,
            ny=nx,
        )
        edge = np.zeros(eq.R.shape, dtype=bool)
        edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True

        reference = boundary_psi_reference(eq.R[edge], eq.Z[edge])

        psi = np.zeros(eq.R.shape)
        freeBoundaryHagenow(eq, blob_current(eq.R, eq.Z), psi)

        errors.append(np.max(np.abs(psi[edge] - reference)) / np.ptp(reference))

    for coarse, fine in pairwise(errors):
        assert np.log2(coarse / fine) > 3.5
    assert errors[-1] < 1e-7
