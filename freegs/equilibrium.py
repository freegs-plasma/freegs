"""
Defines class to represent the equilibrium
state, including plasma and coil currents
"""

from numpy import meshgrid, linspace, exp, zeros, nditer, array
from scipy import interpolate
from scipy.integrate import romb, quad # Romberg integration

from .boundary import fixedBoundary, freeBoundary
from .critical import find_separatrix

# Operators which define the G-S equation
from .gradshafranov import mu0, GSsparse

# Multigrid solver
from . import multigrid

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

    _applyBoundary()
    _solver - Grad-Shafranov elliptic solver

    """

    def __init__(self, tokamak=machine.EmptyTokamak(),
                 Rmin=0.1, Rmax=2.0,
                 Zmin=-1.0, Zmax=1.0,
                 nx=65, ny=65,
                 boundary=freeBoundary,
                 psi=None, current=0.0):
        """Initialises a plasma equilibrium

        Rmin, Rmax  - Range of major radius R [m]
        Zmin, Zmax  - Range of height Z [m]

        nx - Resolution in R. This must be 2^n + 1
        ny - Resolution in Z. This must be 2^m + 1

        boundary - The boundary condition, either freeBoundary or fixedBoundary

        psi - Magnetic flux. If None, use concentric circular flux
              surfaces as starting guess

        current - Plasma current (default = 0.0)
        """

        self.tokamak = tokamak

        self._applyBoundary = boundary

        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Zmin = Zmin
        self.Zmax = Zmax

        self.R_1D = linspace(Rmin, Rmax, nx)
        self.Z_1D = linspace(Zmin, Zmax, ny)
        self.R, self.Z = meshgrid(self.R_1D, self.Z_1D, indexing='ij')

        yymid = 1 - Zmax/(Zmax - Zmin)

        if psi is None:
            # Starting guess for psi
            xx, yy = meshgrid(linspace(0, 1, nx), linspace(0, 1, ny), indexing='ij')
            psi = exp(-((xx - 0.5)**2 + (yy - yymid)**2) / 0.4**2)

            psi[0, :] = 0.0
            psi[:, 0] = 0.0
            psi[-1, :] = 0.0
            psi[:, -1] = 0.0

        self._updatePlasmaPsi(psi)

        # Calculate coil Greens functions. This is an optimisation,
        # used in self.psi() to speed up calculations
        self._pgreen = tokamak.createPsiGreens(self.R, self.Z)

        self._current = current  # Plasma current

        # Create the solver
        generator = GSsparse(Rmin, Rmax, Zmin, Zmax)
        self._solver = multigrid.createVcycle(nx, ny,
                                              generator,
                                              nlevels=1,
                                              ncycle=1,
                                              niter=2,
                                              direct=True)
        
    def setSolverVcycle(nlevels=1, ncycle=1, niter=1, direct=True):
        """
        Creates a new linear solver, based on the multigrid code
    
        nlevels  - Number of resolution levels, including original
        ncycle   - The number of V cycles
        niter    - Number of linear solver (Jacobi) iterations per level
        direct   - Use a direct solver at the coarsest level?
        
        """
        generator = GSsparse(Rmin, Rmax, Zmin, Zmax)
        nx,ny = self.R.shape
        
        self._solver = multigrid.createVcycle(nx, ny,
                                              generator,
                                              nlevels=nlevels,
                                              ncycle=ncycle,
                                              niter=niter,
                                              direct=direct)
        
    def setSolver(solver):
        """
        Sets the linear solver to use. The given object/function must have a __call__ method
        which takes two inputs

        solver(x, b)
        
        where x is the initial guess. This should solve Ax = b, returning the result.

        """
        self._solver = solver

    def callSolver(self, psi, rhs):
        """
        Calls the psi solver, passing the initial guess and RHS arrays
        
        psi   Initial guess for the solution (used if iterative)
        rhs   
        
        Returns
        -------
        
        Solution psi

        """
        return self._solver(psi, rhs)
                        
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
        #return self.plasma_psi + self.tokamak.psi(self.R, self.Z)
        return self.plasma_psi + self.tokamak.calcPsiFromGreens(self._pgreen)
        
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

    def separatrix(self, ntheta=20):
        """
        Returns an array of ntheta (R, Z) coordinates of the separatrix,
        equally spaced in geometric poloidal angle.
        """
        return array(find_separatrix(self, ntheta=ntheta, psi=self.psi()))[:, 0:2]

    def solve(self, profiles):
        """
        Calculate the plasma equilibrium given new profiles
        replacing the current equilibrium.
        
        This performs the linear Grad-Shafranov solve
        
        profiles  - An object describing the plasma profiles.
                    At minimum this must have methods:
             .Jtor(R, Z, psi)   -> [nx, ny]
             .pprime(psinorm)
             .ffprime(psinorm)
             .pressure(psinorm)
             .fpol(psinorm)
        """
        
        self._profiles = profiles
        
        # Calculate toroidal current density
        Jtor = profiles.Jtor(self.R, self.Z, self.psi())
        
        # Set plasma boundary
        # Note that the Equilibrium is passed to the boundary function
        # since the boundary may need to run the G-S solver (von Hagenow's method)
        self._applyBoundary(self, Jtor, self.plasma_psi)
        
        # Right hand side of G-S equation
        rhs = -mu0 * self.R * Jtor

        # Copy boundary conditions
        rhs[0,:] = self.plasma_psi[0,:]
        rhs[:,0] = self.plasma_psi[:,0]
        rhs[-1,:] = self.plasma_psi[-1,:]
        rhs[:,-1] = self.plasma_psi[:,-1]
        
        # Call elliptic solver
        plasma_psi = self._solver(self.plasma_psi, rhs)
        
        self._updatePlasmaPsi(plasma_psi)

        # Update plasma current
        dR = self.R[1,0] - self.R[0,0]
        dZ = self.Z[0,1] - self.Z[0,0]
        self._current = romb(romb(Jtor)) * dR*dZ
        
    def _updatePlasmaPsi(self, plasma_psi):
        """
        Sets the plasma psi data, updates spline interpolation coefficients
        """
        self.plasma_psi = plasma_psi

        # Update spline interpolation
        self.psi_func = interpolate.RectBivariateSpline(self.R[:,0], self.Z[0,:], plasma_psi)
     
    def plot(self, axis=None, show=True, oxpoints=True):
        """
        Plot the equilibrium flux surfaces
        
        axis     - Specify the axis on which to plot
        show     - Call matplotlib.pyplot.show() before returning
        oxpoints - Plot X points as red circles, O points as green circles
    
        """
        from .plotting import plotEquilibrium
        return plotEquilibrium(self, axis=axis, show=show, oxpoints=oxpoints)
    

def refine(eq):
    """
    Double grid resolution, returning a new equilibrium
    
    """
    # Interpolate the plasma psi
    plasma_psi = multigrid.interpolate(eq.plasma_psi)
    nx, ny = plasma_psi.shape
    
    result = Equilibrium(tokamak=eq.tokamak,
                         Rmin = eq.Rmin,
                         Rmax = eq.Rmax,
                         Zmin = eq.Zmin,
                         Zmax = eq.Zmax,
                         nx=nx, ny=ny)

    result._updatePlasmaPsi(plasma_psi)
    
    if hasattr(eq, "_profiles"):
        result._profiles = eq._profiles

    if hasattr(eq, "control"):
        result.control = eq.control

    return result

def coarsen(eq):
    """
    Reduce grid resolution, returning a new equilibrium
    """
    plasma_psi = multigrid.restrict(eq.plasma_psi)
    nx, ny = plasma_psi.shape
    
    result = Equilibrium(tokamak=eq.tokamak,
                         Rmin = eq.Rmin,
                         Rmax = eq.Rmax,
                         Zmin = eq.Zmin,
                         Zmax = eq.Zmax,
                         nx=nx, ny=ny)

    result._updatePlasmaPsi(plasma_psi)
    
    if hasattr(eq, "_profiles"):
        result._profiles = eq._profiles

    if hasattr(eq, "control"):
        result.control = eq.control

    return result

def newDomain(eq,
              Rmin=None, Rmax=None,
              Zmin=None, Zmax=None,
              nx=None, ny=None):
    """Creates a new Equilibrium, solving in a different domain.
    The domain size (Rmin, Rmax, Zmin, Zmax) and resolution (nx,ny)
    are taken from the input equilibrium eq if not specified.
    """
    if Rmin is None:
        Rmin = eq.Rmin
    if Rmax is None:
        Rmax = eq.Rmax
    if Zmin is None:
        Zmin = eq.Zmin
    if Zmax is None:
        Zmax = eq.Zmax
    if nx is None:
        nx = eq.R.shape[0]
    if ny is None:
        ny = eq.R.shape[0]

    # Create a new equilibrium with the new domain
    result = Equilibrium(tokamak=eq.tokamak,
                         Rmin = Rmin,
                         Rmax = Rmax,
                         Zmin = Zmin,
                         Zmax = Zmax,
                         nx=nx, ny=ny)

    # Calculate the current on the old grid
    profiles = eq._profiles
    Jtor = profiles.Jtor(eq.R, eq.Z, eq.psi())

    # Interpolate Jtor onto new grid
    Jtor_func = interpolate.RectBivariateSpline(eq.R[:,0], eq.Z[0,:], Jtor)
    Jtor_new = Jtor_func(result.R, result.Z, grid=False)

    result._applyBoundary(result, Jtor_new, result.plasma_psi)

    # Right hand side of G-S equation
    rhs = -mu0 * result.R * Jtor_new

    # Copy boundary conditions
    rhs[0,:] = result.plasma_psi[0,:]
    rhs[:,0] = result.plasma_psi[:,0]
    rhs[-1,:] = result.plasma_psi[-1,:]
    rhs[:,-1] = result.plasma_psi[:,-1]

    # Call elliptic solver
    plasma_psi = result._solver(result.plasma_psi, rhs)
        
    result._updatePlasmaPsi(plasma_psi)

    # Solve once more, calculating Jtor using new psi
    result.solve(profiles)
    
    return result


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
