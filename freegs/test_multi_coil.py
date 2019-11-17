from .multi_coil import MultiCoil
from .coil import Coil
from .machine import Circuit

import numpy as np

def test_single():
    """
    Test that a single turn MultiCoil is the same as a Coil
    """
    
    mcoil = MultiCoil(1.1, 0.2, current = 100., mirror=False)
    coil = MultiCoil(1.1, 0.2, current = 100.)

    assert np.isclose(coil.controlPsi(0.3, 0.1),
                     mcoil.controlPsi(0.3, 0.1))

    assert np.isclose(coil.controlBr(0.3, 0.1),
                     mcoil.controlBr(0.3, 0.1))

def test_two_turns():
    """
    MultiCoil with two turns same as circuit with two coils
    """

    mcoil = MultiCoil([1.1, 1.2], [0.2, -0.3], current = 100., mirror=False)

    circuit = Circuit([("A", Coil(1.1, 0.2, current=100.), 1.0),
                       ("B", Coil(1.2, -0.3, current=100.), 1.0)])
    
    assert np.isclose(circuit.controlPsi(0.3, 0.1),
                      mcoil.controlPsi(0.3, 0.1))

    assert np.isclose(circuit.controlBr(0.3, 0.1),
                      mcoil.controlBr(0.3, 0.1))


def test_mirrored():
    """
    Mirrored MultiCoil the same as two coils in a circuit
    """
    mcoil = MultiCoil(1.1, 0.2, current = 100., mirror=True)

    circuit = Circuit([("A", Coil(1.1, 0.2, current=100.), 1.0),
                       ("B", Coil(1.1, -0.2, current=100.), 1.0)])
    
    assert np.isclose(circuit.controlPsi(0.3, 0.1),
                      mcoil.controlPsi(0.3, 0.1))

    assert np.isclose(circuit.controlBr(0.3, 0.1),
                      mcoil.controlBr(0.3, 0.1))
