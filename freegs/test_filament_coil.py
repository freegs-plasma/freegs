from .filament_coil import FilamentCoil
from .coil import Coil
from .machine import Circuit

import numpy as np


def test_single():
    """
    Test that a single turn FilamentCoil is the same as a Coil
    """

    fcoil = FilamentCoil(1.1, 0.2, current=100.0)
    coil = FilamentCoil(1.1, 0.2, current=100.0)

    assert np.isclose(coil.controlPsi(0.3, 0.1), fcoil.controlPsi(0.3, 0.1))

    assert np.isclose(coil.controlBr(0.3, 0.1), fcoil.controlBr(0.3, 0.1))


def test_two_turns():
    """
    FilamentCoil with two turns same as circuit with two coils at half current
    """

    fcoil = FilamentCoil([1.1, 1.2], [0.2, -0.3], current=100.0)

    circuit = Circuit(
        [
            ("A", Coil(1.1, 0.2, current=100.0/2.), 1.0),
            ("B", Coil(1.2, -0.3, current=100.0/2.), 1.0),
        ]
    )

    assert np.isclose(circuit.controlPsi(0.3, 0.1), fcoil.controlPsi(0.3, 0.1))

    assert np.isclose(circuit.controlBr(0.3, 0.1), fcoil.controlBr(0.3, 0.1))

    assert np.isclose(circuit.controlBz(0.3, 0.1), fcoil.controlBz(0.3, 0.1))


def test_mirrored():
    """
    Mirrored FilamentCoil the same as two coils in a circuit at half current
    """
    fcoil = FilamentCoil([1.1, 1.1], [0.2, -0.2], current=100.0)

    circuit = Circuit(
        [
            ("A", Coil(1.1, 0.2, current=100.0/2.), 1.0),
            ("B", Coil(1.1, -0.2, current=100.0/2.), 1.0),
        ]
    )

    assert np.isclose(circuit.controlPsi(0.3, 0.1), fcoil.controlPsi(0.3, 0.1))

    assert np.isclose(circuit.controlBr(0.3, 0.1), fcoil.controlBr(0.3, 0.1))

    assert np.isclose(circuit.controlBz(0.3, 0.1), fcoil.controlBz(0.3, 0.1))


def test_move_R():
    """
    Changing major radius property R changes all filament locations
    """

    dR = 0.6
    coil1 = FilamentCoil([1.1, 0.2], [1.2, -0.3], current=100.0)
    coil2 = FilamentCoil([1.1 + dR, 0.2 + dR], [1.2, -0.3], current=100.0)

    # Shift coil1 to same location as coil2
    coil1.R += dR

    assert np.isclose(coil1.controlPsi(0.4, 0.5), coil2.controlPsi(0.4, 0.5))

    assert np.isclose(coil1.controlBr(0.3, -0.2), coil2.controlBr(0.3, -0.2))

    assert np.isclose(coil1.controlBz(1.75, 1.2), coil2.controlBz(1.75, 1.2))


def test_move_Z():
    """
    Changing height property Z changes all filament locations
    """

    dZ = 0.4
    coil1 = FilamentCoil([1.1, 0.2], [1.2, -0.3], current=100.0)
    coil2 = FilamentCoil([1.1, 0.2], [1.2 + dZ, -0.3 + dZ], current=100.0)

    # Shift coil1 to same location as coil2
    coil1.Z += dZ

    assert np.isclose(coil1.controlPsi(0.4, 0.5), coil2.controlPsi(0.4, 0.5))

    assert np.isclose(coil1.controlBr(0.3, -0.2), coil2.controlBr(0.3, -0.2))

    assert np.isclose(coil1.controlBz(1.75, 1.2), coil2.controlBz(1.75, 1.2))
