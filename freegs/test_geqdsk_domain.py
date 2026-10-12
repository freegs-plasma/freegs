import io

import numpy as np
import pytest

import freegs
from freegs import geqdsk


@pytest.fixture(scope="module")
def geqdsk_text():
    eq = freegs.Equilibrium(
        tokamak=freegs.machine.TestTokamak(),
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-1.0,
        Zmax=1.0,
        nx=33,
        ny=33,
        boundary=freegs.boundary.freeBoundaryHagenow,
    )
    profiles = freegs.jtor.ConstrainPaxisIp(eq, 1e3, 2e5, 2.0)
    constrain = freegs.control.constrain(
        xpoints=[(1.1, -0.6), (1.1, 0.8)], isoflux=[(1.1, -0.6, 1.1, 0.6)]
    )
    freegs.solve(eq, profiles, constrain, rtol=1e-4)
    buffer = io.StringIO()
    geqdsk.write(eq, buffer)
    return buffer.getvalue(), eq


@pytest.mark.parametrize(
    "domain",
    [
        (0.2, 1.9, -0.9, 0.9),  # smaller than the input: same resolution
        (0.05, 2.1, -1.1, 1.1),  # larger than the input: finer grid
    ],
)
def test_read_with_domain(geqdsk_text, domain):
    """geqdsk.read(domain=...) returns an equilibrium on the requested domain"""
    text, original = geqdsk_text
    eq = geqdsk.read(
        io.StringIO(text), freegs.machine.TestTokamak(), domain=list(domain), rtol=1e-3
    )

    assert np.isclose(eq.Rmin, domain[0])
    assert np.isclose(eq.Rmax, domain[1])
    assert np.isclose(eq.Zmin, domain[2])
    assert np.isclose(eq.Zmax, domain[3])
    assert np.isclose(eq.plasmaCurrent(), original.plasmaCurrent(), rtol=0.05)
