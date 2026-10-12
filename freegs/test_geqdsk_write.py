import io

import numpy as np
from freeqdsk import geqdsk as freeqdsk_geqdsk

import freegs
from freegs import geqdsk


def test_write_profiles_on_axis_to_boundary_grid():
    """
    The G-EQDSK 1D profiles are on a uniform grid in psi from the axis to the
    boundary, both included, so that entry i is at psiN = i / (nx - 1). This is
    the grid used by freeqdsk and by geqdsk.read.
    """
    tokamak = freegs.machine.TestTokamak()
    eq = freegs.Equilibrium(
        tokamak=tokamak,
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-1.0,
        Zmax=1.0,
        nx=17,
        ny=17,
        boundary=freegs.boundary.freeBoundaryHagenow,
    )
    profiles = freegs.jtor.ConstrainPaxisIp(eq, 1e4, 1e6, 2.0)
    constrain = freegs.control.constrain(
        xpoints=[(1.1, -0.6), (1.1, 0.8)], isoflux=[(1.1, -0.6, 1.1, 0.6)]
    )
    freegs.solve(eq, profiles, constrain, maxits=25, atol=1e-3, rtol=1e-1)

    buffer = io.StringIO()
    geqdsk.write(eq, buffer)
    buffer.seek(0)
    data = freeqdsk_geqdsk.read(buffer)

    psinorm = np.linspace(0.0, 1.0, data["nx"])
    assert np.allclose(data["pprime"], eq.pprime(psinorm))
    assert np.allclose(data["pres"], eq.pressure(psinorm))
    assert np.allclose(data["fpol"], eq.fpol(psinorm))
    # The last entry is the boundary value
    assert np.isclose(data["pres"][-1], eq.pressure(1.0))
