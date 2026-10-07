"""Safety-factor oracle for nested surfaces without an X-point."""
from types import SimpleNamespace
import numpy as np
import pytest
from .equilibrium import Equilibrium
from .boundary import fixedBoundary


def circular_equilibrium():
    R0, a, C, F = 6.2, .62, 1.7, 32.86
    r = np.linspace(R0-1.25*a,R0+1.25*a,65)
    z = np.linspace(-1.25*a,1.25*a,65)
    R,Z = np.meshgrid(r,z,indexing="ij")
    psi = C*((R-R0)**2+Z**2)
    eq = Equilibrium(Rmin=r[0],Rmax=r[-1],Zmin=z[0],Zmax=z[-1],
                     nx=65,ny=65,psi=psi,boundary=fixedBoundary)
    eq.psi_bndry = C*a*a
    eq._profiles = SimpleNamespace(fpol=lambda s:np.full_like(s,F))
    return eq,R0,a,C,F


def test_q_without_xpoint_matches_circular_line_integral():
    eq,R0,a,C,F = circular_equilibrium()
    s = np.array([.2,.5,.8])
    # q = (F/(2 C)) <1/R> around each circular surface.
    expected = F/(2*C*np.sqrt(R0*R0-s*a*a))
    np.testing.assert_allclose(eq.q(s),expected,rtol=1e-3)


def test_q_without_xpoint_requires_known_boundary_flux():
    eq,*_ = circular_equilibrium()
    eq.psi_bndry = None
    with pytest.raises(ValueError,match="boundary"):
        eq.q(np.array([.5]))
