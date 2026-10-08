"""Exact ellipse/coarea oracle independent of ray/tangent algorithms.

Supply the declared contour-edge flux through the existing xpoint metadata
argument. Its geometric marker lies on the ellipse; this is not a critical-point
finder or physical-separatrix test. Fixed-edge fallback is separate PR1.
"""
from types import SimpleNamespace
import unittest
import numpy as np
from .critical import find_safety


def ellipse_case(kappa=1.0, alpha=0.0, f=32.86, domain_ratio=1.0):
    r0,c,a=6.2,2.0,.62
    r,z=np.meshgrid(np.linspace(r0-1.8*a,r0+1.8*a,65),
                    np.linspace(-1.8*a*domain_ratio,1.8*a*domain_ratio,65),indexing='ij')
    def coordinates(r,z):
        x=r-r0
        return x*np.cos(alpha)+z*np.sin(alpha),-x*np.sin(alpha)+z*np.cos(alpha)
    def psi(r,z):
        x,y=coordinates(r,z)
        return c*(x*x+y*y/kappa**2)
    def gradient(r,z):
        x,y=coordinates(r,z)
        return (2*c*(x*np.cos(alpha)-y*np.sin(alpha)/kappa**2),
                2*c*(x*np.sin(alpha)+y*np.cos(alpha)/kappa**2))
    eq=SimpleNamespace(R=r,Z=z,Rmin=r.min(),Rmax=r.max(),Zmin=z.min(),Zmax=z.max(),
        psi_bndry=c*a*a,psi=lambda:psi(r,z),fpol=lambda s:np.zeros_like(s)+f,
        Br=lambda r,z:-gradient(r,z)[1]/r,Bz=lambda r,z:gradient(r,z)[0]/r)
    eq.q_reference=[(r0+a*np.cos(alpha),a*np.sin(alpha),eq.psi_bndry)]
    def exact(s):
        amplitude2=a*a*np.asarray(s)*(np.cos(alpha)**2+kappa**2*np.sin(alpha)**2)
        return f*kappa/(2*c*np.sqrt(r0*r0-amplitude2))
    return eq,exact,[(r0,0.0,0.0)]


class SafetyContourOracle(unittest.TestCase):
    def test_circle_signed_exact(self):
        for f in [32.86,-32.86]:
            eq,exact,axis=ellipse_case(f=f)
            surfaces=np.array([.01,.1,.5,.9])
            actual=find_safety(eq,psinorm=surfaces,ntheta=128,opoint=axis,xpoint=eq.q_reference)
            np.testing.assert_allclose(actual,exact(surfaces),rtol=1e-11,atol=1e-12)

    def test_rotated_ellipse_exact(self):
        eq,exact,axis=ellipse_case(kappa=1.7,alpha=.31)
        surfaces=np.array([.01,.1,.5,.9])
        actual=find_safety(eq,psinorm=surfaces,ntheta=128,opoint=axis,xpoint=eq.q_reference)
        np.testing.assert_allclose(actual,exact(surfaces),rtol=1e-10,atol=1e-11)

    def test_ellipse_angular_refinement(self):
        eq,exact,axis=ellipse_case(kappa=1.7,alpha=.31)
        surfaces=np.array([.1,.5,.9]);errors=[]
        for n in [16,32,64,128]:
            actual=find_safety(eq,psinorm=surfaces,ntheta=n,opoint=axis,xpoint=eq.q_reference)
            errors.append(np.max(np.abs(actual/exact(surfaces)-1)))
        self.assertGreater(errors[0],1e-6)
        self.assertLess(errors[-1],1e-10)
        self.assertLess(errors[-1],errors[0]/1000)

    def test_rotated_ellipse_rectangular_domain(self):
        eq,exact,axis=ellipse_case(kappa=1.7,alpha=.31,domain_ratio=1.6)
        surfaces=np.array([.01,.1,.5,.9])
        actual=find_safety(eq,psinorm=surfaces,ntheta=128,opoint=axis,xpoint=eq.q_reference)
        np.testing.assert_allclose(actual,exact(surfaces),rtol=1e-10,atol=1e-11)


if __name__=='__main__':unittest.main()
