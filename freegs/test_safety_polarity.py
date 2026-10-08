"""Physical winding on an analytic O/X-point field, independent of contour code."""
from types import SimpleNamespace
import unittest
import numpy as np
from .critical import find_safety


class SignedSafetyOracle(unittest.TestCase):
    def test_physical_winding_all_poloidal_and_toroidal_polarities(self):
        r0, a = 6.2, .62
        r, z = np.meshgrid(np.linspace(r0-1.5*a, r0+1.5*a,129),
                           np.linspace(-1.5*a,1.5*a,129),indexing='ij')
        theta = np.arange(8192)*2*np.pi/8192
        levels = np.array([.1,.5,.9])
        for c in [2.,-2.]:
            for f in [32.86,-32.86]:
                # O-point (r0,0), genuine hyperbolic critical points at
                # (r0,+/-a/sqrt(2)), flux c*a^2/4 and opposite Hessian signs.
                flux = c*((r-r0)**2+z*z-z**4/a**2)
                eq = SimpleNamespace(R=r,Z=z,Rmin=r.min(),Rmax=r.max(),
                    Zmin=z.min(),Zmax=z.max(),psi=lambda:flux,
                    fpol=lambda s:np.zeros_like(s)+f,
                    Br=lambda rr,zz:-c*(2*zz-4*zz**3/a**2)/rr,
                    Bz=lambda rr,zz:2*c*(rr-r0)/rr)
                # Exact inner radial root of u-u^2*sin(theta)^4/a^2=s*a^2/4.
                target = levels[:,None]*a*a/4
                u = 2*target/(1+np.sqrt(1-4*target*np.sin(theta)**4/a**2))
                rr = r0+np.sqrt(u)*np.cos(theta)
                zz = np.sqrt(u)*np.sin(theta)
                theta_rate = (eq.Bz(rr,zz)*(rr-r0)-eq.Br(rr,zz)*zz)/u
                physical_q = np.mean((f/rr**2)/theta_rate,axis=1)
                actual = find_safety(eq,psinorm=levels,ntheta=512,
                    opoint=[(r0,0.,0.)],xpoint=[(r0,a/np.sqrt(2),c*a*a/4)])
                # Existing chord/root/bicubic errors are separate from this
                # polarity fix; no contour/tangent or fixed-edge PR is needed.
                np.testing.assert_allclose(actual,physical_q,rtol=2e-3,atol=1e-12)

    def test_invalid_flux_span_rejected(self):
        r,z=np.meshgrid(np.linspace(5,7,9),np.linspace(-1,1,9),indexing='ij')
        eq=SimpleNamespace(R=r,Z=z,psi=lambda:(r-6)**2+z*z)
        for span in [0.,np.inf,np.nan]:
            with self.assertRaisesRegex(ValueError,'finite nonzero'):
                find_safety(eq,opoint=[(6.,0.,0.)],xpoint=[(6.,1.,span)])
