"""
Tests of the coil current control system
"""

import numpy as np
import pytest

from . import control, jtor, machine
from .equilibrium import Equilibrium
from .picard import solve


@pytest.fixture(scope="module")
def diverted_equilibrium():
    tokamak = machine.TestTokamak()
    eq = Equilibrium(
        tokamak=tokamak, Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0, nx=33, ny=33
    )
    profiles = jtor.ConstrainPaxisIp(eq, 1e3, 2e5, 2.0)
    constrain = control.constrain(
        xpoints=[(1.1, -0.6), (1.1, 0.8)], isoflux=[(1.1, -0.6, 1.1, 0.6)]
    )
    solve(eq, profiles, constrain, rtol=1e-4)
    return eq


def test_advanced_constraints_default_arguments(diverted_equilibrium):
    """
    The advanced 2D constraints can be used without current limits and with the
    default scalar weights. Fitting the equilibrium's own psi must not change
    the coil currents.
    """
    eq = diverted_equilibrium
    tokamak = eq.getMachine()
    currents = np.array(tokamak.controlCurrents())

    try:
        for constraint in (
            control.ConstrainPsi2DAdvanced(eq.psi()),
            control.ConstrainPsiNorm2DAdvanced(eq.psiN()),
        ):
            constraint(eq)
            assert np.allclose(tokamak.controlCurrents(), currents, rtol=1e-3, atol=1.0)
            tokamak.setControlCurrents(currents)
    finally:
        tokamak.setControlCurrents(currents)
