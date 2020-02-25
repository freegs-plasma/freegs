import freegs

import io
from numpy import allclose


def test_readwrite():
    """Test reading/writing to a file round-trip
    """
    tokamak = freegs.machine.MAST_sym()
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
    profiles = freegs.jtor.ConstrainPaxisIp(1e4, 1e6, 2.0)
    xpoints = [(1.1, -0.6), (1.1, 0.8)]
    isoflux = [(1.1, -0.6, 1.1, 0.6)]
    constrain = freegs.control.constrain(xpoints=xpoints, isoflux=isoflux)
    freegs.solve(eq, profiles, constrain, maxits=25, atol=1e-3, rtol=1e-1)

    memory_file = io.BytesIO()

    with freegs.OutputFile(memory_file, "w") as f:
        f.write_equilibrium(eq)

    with freegs.OutputFile(memory_file, "r") as f:
        read_eq = f.read_equilibrium()

    assert tokamak == read_eq.tokamak
    assert allclose(eq.psi(), read_eq.psi())
