from . import equilibrium

import numpy as np

def test_inoutseparatrix():

    eq = equilibrium.Equilibrium(Rmin=0.1, Rmax=2.0,
                                 Zmin=-1.0, Zmax=1.0,
                                 nx=65, ny=65)
    
    # Two O-points, one X-point half way between them
    psi = (np.exp((-(eq.R - 1.0)**2 - eq.Z**2)*3) +
           np.exp((-(eq.R - 1.0)**2 - (eq.Z + 1)**2)*3))

    eq._updatePlasmaPsi(psi)

    Rin, Rout = eq.innerOuterSeparatrix()

    assert Rin >= eq.Rmin and Rout >= eq.Rmin
    assert Rin <= eq.Rmax and Rout <= eq.Rmax
    
