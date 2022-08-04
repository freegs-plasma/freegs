#
# Test calculation of magnetic field due to coils
# and forces between coils

from . import machine
import numpy as np
import math

mu0 = 4e-7 * np.pi


def test_coil_axis():
    """
    Test magnetic field due to a single coil, on axis
    """

    Rcoil = 1.5
    current = 123.0
    coil = machine.Coil(Rcoil, 1.0, current=current)

    def analytic_Bz(dZ):
        return (mu0 / 2) * Rcoil ** 2 * current / (dZ ** 2 + Rcoil ** 2) ** 1.5

    # Note: Can't evaluate at R=0,
    assert math.isclose(coil.Br(0.0001, 2.0), 0.0, abs_tol=1e-8)
    assert math.isclose(coil.Bz(0.001, 2.0), analytic_Bz(1.0), abs_tol=1e-8)
    assert math.isclose(coil.Bz(0.001, -1.0), analytic_Bz(-2.0), abs_tol=1e-8)


def test_coil_forces():
    """
    Test forces between two coils
    """
    Rcoil = 1.5
    current = 123.0

    coil1 = machine.Coil(Rcoil, 1.0, current=current)
    coil2 = machine.Coil(Rcoil, -1.0, current=current)

    tokamak = machine.Machine([("P1", coil1), ("P2", coil2)])

    forces = tokamak.getForces()

    assert "P1" in forces and "P2" in forces

    # Vertical force is equal and opposite
    assert math.isclose(forces["P1"][1], -forces["P2"][1])
    assert forces["P1"][1] < 0.0  # Force downward towards other coil

    # Reverse one of the currents
    coil2.current = -123.0
    # check the force reverses direction
    forces = tokamak.getForces()
    assert forces["P1"][1] > 0.0

    print(forces)


def test_coil_forces_unequal():
    """
    Test forces between two different sized coils
    """

    coil1 = machine.Coil(1.5, 1.0, current=123)
    coil2 = machine.Coil(2.5, -1.0, current=456)

    tokamak = machine.Machine([("P1", coil1), ("P2", coil2)])

    forces = tokamak.getForces()

    assert "P1" in forces and "P2" in forces

    # Vertical force is equal and opposite
    assert math.isclose(forces["P1"][1], -forces["P2"][1])
    assert forces["P1"][1] < 0.0  # Force downward towards other coil
