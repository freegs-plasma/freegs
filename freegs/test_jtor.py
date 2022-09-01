import numpy as np

from . import jtor
from . import equilibrium


def test_psinorm_range():
    """Test that the profiles produce finite values outside core"""

    eq = equilibrium.Equilibrium(Rmin = 0.5,
                                 Rmax = 1.5,
                                 Zmin = -1.0,
                                 Zmax = 1.0,
                                 nx = 33,
                                 ny = 33)

    for profiles in [
        jtor.ConstrainPaxisIp(eq, 1e3, 2e5, 2.0),
        jtor.ConstrainBetapIp(eq, 1.0, 2e5, 2.0),
    ]:
        current_density = profiles.Jtor(eq.R, eq.Z, eq.psi())
        assert np.all(np.isfinite(current_density))

        assert profiles.pprime(1.0) == 0.0
        assert profiles.pprime(1.1) == 0.0
        assert np.isfinite(profiles.pprime(-0.32))

        assert profiles.ffprime(1.0) == 0.0
        assert profiles.ffprime(1.1) == 0.0
        assert np.isfinite(profiles.ffprime(-0.32))
