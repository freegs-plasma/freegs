#
# Test calculation of magnetic field due to coils
# and forces between coils

import math

import numpy as np

from . import machine

mu0 = 4e-7 * np.pi


def test_coil_axis():
    """
    Test magnetic field due to a single coil, on axis
    """

    Rcoil = 1.5
    current = 123.0
    coil = machine.Coil(Rcoil, 1.0, current=current)

    def analytic_Bz(dZ):
        return (mu0 / 2) * Rcoil**2 * current / (dZ**2 + Rcoil**2) ** 1.5

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

    assert "P1" in forces
    assert "P2" in forces

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

    assert "P1" in forces
    assert "P2" in forces

    # Vertical force is equal and opposite
    assert math.isclose(forces["P1"][1], -forces["P2"][1])
    assert forces["P1"][1] < 0.0  # Force downward towards other coil


def test_circuit_force_matches_standalone_coil():
    """
    The force on a coil must not depend on whether it is wrapped in a
    one-coil Circuit (the Circuit field evaluation must not restore the
    coil's own self-field)
    """
    standalone = machine.Machine(
        [
            ("A", machine.Coil(1.5, 1.0, current=123.0)),
            ("B", machine.Coil(2.5, -1.0, current=456.0)),
        ]
    )
    circuit = machine.Circuit(
        [("A", machine.Coil(1.5, 1.0, turns=1), 1.0)], current=123.0
    )
    wrapped = machine.Machine(
        [("C", circuit), ("B", machine.Coil(2.5, -1.0, current=456.0))]
    )

    f_standalone = standalone.getForces()["A"]
    f_circuit = wrapped.getForces()["C"]["A"]

    assert np.allclose(f_circuit, f_standalone, rtol=1e-10, atol=0.0)


def test_multi_coil_circuit_force_matches_independent_coils():
    """
    Forces on the coils of a two-coil Circuit match the same coils placed
    individually with their circuit currents (multipliers applied)
    """
    current = 100.0
    circuit = machine.Circuit(
        [
            ("U", machine.Coil(1.5, 1.0), 1.0),
            ("L", machine.Coil(1.5, -1.0), -1.0),
        ],
        current=current,
    )
    wrapped = machine.Machine(
        [("C", circuit), ("X", machine.Coil(2.0, 0.0, current=50.0))]
    )
    independent = machine.Machine(
        [
            ("U", machine.Coil(1.5, 1.0, current=current)),
            ("L", machine.Coil(1.5, -1.0, current=-current)),
            ("X", machine.Coil(2.0, 0.0, current=50.0)),
        ]
    )

    f_circuit = wrapped.getForces()
    f_independent = independent.getForces()

    for label in ("U", "L"):
        assert np.allclose(
            f_circuit["C"][label], f_independent[label], rtol=1e-10, atol=0.0
        )
    assert np.allclose(f_circuit["X"], f_independent["X"], rtol=1e-10, atol=0.0)
