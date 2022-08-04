from .multi_coil import MultiCoil
from .coil import Coil
from .machine import Circuit

import numpy as np


def test_single():
    """
    Test that a single turn MultiCoil is the same as a Coil
    """

    mcoil = MultiCoil(1.1, 0.2, current=100.0, mirror=False)
    coil = MultiCoil(1.1, 0.2, current=100.0)

    assert np.isclose(coil.controlPsi(0.3, 0.1), mcoil.controlPsi(0.3, 0.1))

    assert np.isclose(coil.controlBr(0.3, 0.1), mcoil.controlBr(0.3, 0.1))


def test_two_turns():
    """
    MultiCoil with two turns same as circuit with two coils
    """

    mcoil = MultiCoil([1.1, 1.2], [0.2, -0.3], current=100.0, mirror=False)

    circuit = Circuit(
        [
            ("A", Coil(1.1, 0.2, current=100.0), 1.0),
            ("B", Coil(1.2, -0.3, current=100.0), 1.0),
        ]
    )

    assert np.isclose(circuit.controlPsi(0.3, 0.1), mcoil.controlPsi(0.3, 0.1))

    assert np.isclose(circuit.controlBr(0.3, 0.1), mcoil.controlBr(0.3, 0.1))

    assert np.isclose(circuit.controlBz(0.3, 0.1), mcoil.controlBz(0.3, 0.1))


def test_mirrored():
    """
    Mirrored MultiCoil the same as two coils in a circuit
    """
    mcoil = MultiCoil(1.1, 0.2, current=100.0, mirror=True)

    circuit = Circuit(
        [
            ("A", Coil(1.1, 0.2, current=100.0), 1.0),
            ("B", Coil(1.1, -0.2, current=100.0), 1.0),
        ]
    )

    assert np.isclose(circuit.controlPsi(0.3, 0.1), mcoil.controlPsi(0.3, 0.1))

    assert np.isclose(circuit.controlBr(0.3, 0.1), mcoil.controlBr(0.3, 0.1))

    assert np.isclose(circuit.controlBz(0.3, 0.1), mcoil.controlBz(0.3, 0.1))


def test_move_R():
    """
    Changing major radius property R changes all filament locations
    """

    dR = 0.6
    coil1 = MultiCoil([1.1, 0.2], [1.2, -0.3], current=100.0, mirror=False)
    coil2 = MultiCoil([1.1 + dR, 0.2 + dR], [1.2, -0.3], current=100.0, mirror=False)

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
    coil1 = MultiCoil([1.1, 0.2], [1.2, -0.3], current=100.0, mirror=False)
    coil2 = MultiCoil([1.1, 0.2], [1.2 + dZ, -0.3 + dZ], current=100.0, mirror=False)

    # Shift coil1 to same location as coil2
    coil1.Z += dZ

    assert np.isclose(coil1.controlPsi(0.4, 0.5), coil2.controlPsi(0.4, 0.5))

    assert np.isclose(coil1.controlBr(0.3, -0.2), coil2.controlBr(0.3, -0.2))

    assert np.isclose(coil1.controlBz(1.75, 1.2), coil2.controlBz(1.75, 1.2))
