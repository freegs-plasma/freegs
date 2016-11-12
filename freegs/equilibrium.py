"""
Defines class to represent the equilibrium
state, including plasma and coil currents
"""

from numpy import meshgrid, linspace, exp, zeros, nditer
from scipy import interpolate
from scipy.integrate import romb, quad # Romberg integration

from .boundary import fixedBoundary, freeBoundary

# Operators which define the G-S equation
from .gradshafranov import mu0, GSElliptic

# Multigrid solver
from .multigrid import smoothMG

from . import machine

class Equilibrium:
    """
    Represents the equilibrium state, including
    plasma and coil currents
    
    Data members
    ------------
    
    These can be read, but should not be modified directly
    
    R[nx,ny]
    Z[nx,ny] 
    
    Rmin, Rmax
    Zmin, Zmax

    tokamak - The coils and circuits
    
    Private data members

    _GS     Grad-Shafranov operator
    _applyBoundary()
    
    """
    
    def __init__(self, tokamak=machine.EmptyTokamak(), 
                 Rmin=0.1, Rmax=2.0,
                 Zmin=-1.0, Zmax=1.0,
                 nx=65, ny=65,
                 boundary=freeBoundary):
        """
        Initialises a plasma equilibrium
        
        Rmin, Rmax  - Range of major radius R [m]
        Zmin, Zmax  - Range of height Z [m]
        
        nx - Resolution in R. This must be 2^n + 1
        ny - Resolution in Z. This must be 2^m + 1
        
        boundary - The boundary condition, either freeBoundary or fixedBoundary
        
        """

        self.tokamak = tokamak

        self._GS = GSElliptic(Rmin)

        self._applyBoundary = boundary
        
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Zmin = Zmin
        self.Zmax = Zmax
        
        dR = (Rmax - Rmin)/(nx - 1)
        dZ = (Zmax - Zmin)/(ny - 1)
        
        # Starting guess for psi
        xx, yy = meshgrid(linspace(0,1,nx), linspace(0,1,ny), indexing='ij')
        psi = exp( - ( (xx - 0.5)**2 + (yy - 0.5)**2 ) / 0.4**2 )
        
        psi[0,:] = 0.0
        psi[:,0] = 0.0
        psi[-1,:] = 0.0
        psi[:,-1] = 0.0
        
        self.R = zeros([nx,ny])
        for x in range(nx):
            self.R[x,:] = Rmin + x * dR
            
        self.Z = zeros([nx,ny])
        for y in range(ny):
            self.Z[:,y] = Zmin + y*dZ
            
        self._updatePlasmaPsi(psi)
        
        self._current = 0.0 # Plasma current
        
    def getMachine(self):
        """
        Returns the handle of the machine, including coils
        """
        return self.tokamak
    
    def plasmaCurrent(self):
        """
        Plasma current [Amps]
        """
        return self._current
    
    def plasmaBr(self, R,Z):
        """
        Radial magnetic field due to plasma
        Br = -1/R dpsi/dZ
        """
        return -self.psi_func(R,Z,dy=1)[0][0]/R
        
    def plasmaBz(self, R, Z):
        """
        Vertical magnetic field due to plasma 
        Bz = (1/R) dpsi/dR
        """
        return self.psi_func(R,Z,dx=1)[0][0]/R
        
    def Br(self, R, Z):
        """
        Total radial magnetic field
        """
        return self.plasmaBr(R,Z) + self.tokamak.Br(R,Z)

    def Bz(self, R, Z):
        """
        Total vertical magnetic field
        """
        return self.plasmaBz(R,Z) + self.tokamak.Bz(R,Z)
    
    def psi(self):
        return self.plasma_psi + self.tokamak.psi(self.R, self.Z)
        
    def psiRZ(self, R, Z):
        """
        Return poloidal flux psi at given (R,Z) location
        """
        return self.psi_func(R,Z) + self.tokamak.psi(R,Z)

    def fpol(self,psinorm):
        """
        Return f = R*Bt at specified values of normalised psi
        """
        return self._profiles.fpol(psinorm)
    
    def fvac(self):
        """
        Return vacuum f = R*Bt
        """
        return self._profiles.fvac()
    
    def q(self, psinorm):
        """
        Returns safety factor q at specified values of normalised psi
        """
        return psinorm * 0.0
        
    def pprime(self, psinorm):
        """
        Return p' at given normalised psi
        """
        return self._profiles.pprime(psinorm)
    
    def ffprime(self, psinorm):
        """
        Return ff' at given normalised psi
        """
        return self._profiles.ffprime(psinorm)
        
    def pressure(self, psinorm, out=None):
        """
        Returns plasma pressure at specified values of normalised psi
        """
        return self._profiles.pressure(psinorm)


    def solve(self, profiles, niter=2, sublevels=4, ncycle=2):
        """
        Calculate the plasma equilibrium
        
        profiles  - An object describing the plasma profiles.
                    At minimum this must have methods:
             .Jtor(R, Z, psi)   -> [nx, ny]
             .pprime(psinorm)
             .ffprime(psinorm)
             .pressure(psinorm)
             .fpol(psinorm)

        niter  - Number of Jacobi iterations per level
        sublevels - Number of levels in the multigrid
        ncycle    - Number of V-cycles
        """
        
        self._profiles = profiles
        
        # Calculate toroidal current density
        Jtor = profiles.Jtor(self.R, self.Z, self.psi())
        
        # Set plasma boundary
        self._applyBoundary(self.R, self.Z, Jtor, self.plasma_psi)
        
        # Right hand side of G-S equation
        rhs = -mu0 * self.R * Jtor
        
        rhs[0,:] = 0.0
        rhs[:,0] = 0.0
        rhs[-1,:] = 0.0
        rhs[:,-1] = 0.0
        
        # Solve elliptic operator using MG method
        dR = self.R[1,0] - self.R[0,0]
        dZ = self.Z[0,1] - self.Z[0,0]
        
        plasma_psi = smoothMG(self._GS, self.plasma_psi, rhs, dR, dZ, 
                              niter=niter, sublevels=sublevels, ncycle=ncycle)
        
        self._updatePlasmaPsi(plasma_psi)

        # Update plasma current
        self._current = romb(romb(Jtor)) * dR*dZ

    def toDict(self):
        """
        Convert to a dictionary, in a format which can be written to file
        
        """
        {"rmin":self.Rmin, "rmax":self.Rmax, 
         "zmin":self.Zmin, "zmax":self.Zmax,
         "psi":self.psi(),
         "plasma_psi":self.plasma_psi()}
         
    def _updatePlasmaPsi(self, plasma_psi):
        """
        Sets the plasma psi data, updates spline interpolation coefficients
        """
        self.plasma_psi = plasma_psi

        # Update spline interpolation
        self.psi_func = interpolate.RectBivariateSpline(self.R[:,0], self.Z[0,:], plasma_psi)
     


def gridup(eq):
    """
    Double grid resolution, returning a new equilibrium
    
    """
    pass


def griddown(eq):
    """
    Reduce grid resolution, returning a new equilibrium
    """
    pass
    

if __name__=="__main__":
    
    # Test the different spline interpolation routines
    
    from numpy import ravel
    import matplotlib.pyplot as plt
    
    import machine
    tokamak = machine.TestTokamak()

    Rmin=0.1
    Rmax=2.0

    eq = Equilibrium(tokamak, Rmin=Rmin, Rmax=Rmax)
    
    import constraints
    xpoints = [(1.2, -0.8),
               (1.2, 0.8)]
    constraints.xpointConstrain(eq, xpoints)
    
    psi = eq.psi()

    tck = interpolate.bisplrep(ravel(eq.R), ravel(eq.Z), ravel(psi))
    spline = interpolate.RectBivariateSpline(eq.R[:,0], eq.Z[0,:], psi)
    f = interpolate.interp2d(eq.R[:,0], eq.Z[0,:],psi, kind='cubic')

    plt.plot(eq.R[:,10], psi[:,10], 'o')

    r = linspace(Rmin, Rmax, 1000)
    z = eq.Z[0,10]
    plt.plot(r, f(r,z), label="f")
    
    plt.plot(r, spline(r,z), label="spline")
    
    plt.plot(r, interpolate.bisplev(r,z, tck), label="bisplev")
    
    plt.legend()
    plt.show()
