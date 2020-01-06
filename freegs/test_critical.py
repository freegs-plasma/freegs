from . import critical

import numpy as np

def test_single_opoint():
    # Create an R-Z grid
    R, Z = np.meshgrid(np.linspace(0.5,1.5,33), np.linspace(-1,1,33), indexing='ij')
    
    # One O-point at (1.0, 0.0) with psi = 1.0
    psi = np.exp((-(R - 1.0)**2 - Z**2)*3)

    opoint, xpoint = critical.find_critical(R, Z, psi)

    assert len(opoint) == 1
    assert len(xpoint) == 0
    np.testing.assert_allclose(opoint[0], [1.0, 0.0, 1.0])
    
def test_single_xpoint():
    # Create an R-Z grid
    R, Z = np.meshgrid(np.linspace(0.5,1.5,65), np.linspace(-1,1,65), indexing='ij')

    # Two O-points, one X-point half way between them
    psi = np.exp((-(R - 1.0)**2 - Z**2)*3) + np.exp((-(R - 1.0)**2 - (Z + 1)**2)*3)
    
    opoint, xpoint = critical.find_critical(R, Z, psi)

    assert len(opoint) == 2
    assert len(xpoint) == 1
    np.testing.assert_allclose(xpoint[0], [1.0, -0.5, 2*np.exp(-3*0.5**2)])
