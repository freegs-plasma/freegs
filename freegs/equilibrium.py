"""
Defines class to represent the equilibrium
state, including plasma and coil currents
"""

from numpy import pi, meshgrid, linspace, exp, zeros, nditer, array
import numpy as np
from scipy import interpolate
from scipy.integrate import romb, quad # Romberg integration

from .boundary import fixedBoundary, freeBoundary
from . import critical

# Operators which define the G-S equation
from .gradshafranov import mu0, GSsparse, GSsparse4thOrder

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
    _profiles     An object which calculates the toroidal current
    _constraints  Control system which adjusts coil currents to meet constraints
                  e.g. X-point location and flux values
    """

    def __init__(self, tokamak=machine.EmptyTokamak(),
                 Rmin=0.1, Rmax=2.0,
                 Zmin=-1.0, Zmax=1.0,
                 nx=65, ny=65,
                 boundary=freeBoundary,
                 psi=None, current=0.0, order=4):
        """Initialises a plasma equilibrium

        Rmin, Rmax  - Range of major radius R [m]
        Zmin, Zmax  - Range of height Z [m]

        nx - Resolution in R. This must be 2^n + 1
        ny - Resolution in Z. This must be 2^m + 1

        boundary - The boundary condition, either freeBoundary or fixedBoundary

        psi - Magnetic flux. If None, use concentric circular flux
              surfaces as starting guess

        current - Plasma current (default = 0.0)

        order - The order of the differential operators to use.
                Valid values are 2 or 4.
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
        
        if psi is None:
            # Starting guess for psi
            xx, yy = meshgrid(linspace(0, 1, nx), linspace(0, 1, ny), indexing='ij')
            psi = exp(-((xx - 0.5)**2 + (yy - 0.5)**2) / 0.4**2)

            psi[0, :] = 0.0
            psi[:, 0] = 0.0
            psi[-1, :] = 0.0
            psi[:, -1] = 0.0

        # Calculate coil Greens functions. This is an optimisation,
        # used in self.psi() to speed up calculations
        self._pgreen = tokamak.createPsiGreens(self.R, self.Z)

        self._current = current  # Plasma current
        
        self._updatePlasmaPsi(psi)  # Needs to be after _pgreen
        
        # Create the solver
        if order == 2:
            generator = GSsparse(Rmin, Rmax, Zmin, Zmax)
        elif order == 4:
            generator = GSsparse4thOrder(Rmin, Rmax, Zmin, Zmax)
        else:
            raise ValueError("Invalid choice of order ({}). Valid values are 2 or 4.".format(order))
        self.order = order
        
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

    def poloidalBeta(self):
        """ 
        Return the poloidal beta 
        betap = (8pi/mu0) * int(p)dRdZ / Ip^2
        """
        
        dR = self.R[1,0] - self.R[0,0]
        dZ = self.Z[0,1] - self.Z[0,0]

        # Normalised psi
        psi_norm = (self.psi() - self.psi_axis)  / (self.psi_bndry - self.psi_axis)

        # Plasma pressure
        pressure = self.pressure(psi_norm)
        if self.mask is not None:
            # If there is a masking function (X-points, limiters)
            pressure *= self.mask

        # Integrate pressure in 2D
        return ((8.*pi)/mu0)*romb(romb(pressure))*dR*dZ / (self.plasmaCurrent()**2)

    def plasmaVolume(self):
        """Calculate the volume of the plasma in m^3"""
        
        dR = self.R[1,0] - self.R[0,0]
        dZ = self.Z[0,1] - self.Z[0,0]

        # Volume element
        dV = 2.*pi*self.R * dR * dZ
        
        if self.mask is not None:   # Only include points in the core
            dV *= self.mask
        
        # Integrate volume in 2D
        return romb(romb(dV))
    
    def plasmaBr(self, R,Z):
        """
        Radial magnetic field due to plasma
        Br = -1/R dpsi/dZ
        """
        return -self.psi_func(R,Z,dy=1, grid=False)/R
        
    def plasmaBz(self, R, Z):
        """
        Vertical magnetic field due to plasma 
        Bz = (1/R) dpsi/dR
        """
        return self.psi_func(R,Z,dx=1, grid=False)/R
        
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

    def Btor(self, R, Z):
        """
        Toroidal magnetic field
        """
        # Normalised psi
        psi_norm = (self.psiRZ(R, Z) - self.psi_axis)  / (self.psi_bndry - self.psi_axis)

        # Get f = R * Btor in the core. May be invalid outside the core
        fpol = self.fpol(psi_norm)
        
        if self.mask is not None:
            # Get the values of the core mask at the requested R,Z locations
            # This is 1 in the core, 0 outside
            mask = self.mask_func(R,Z, grid=False)
            fpol = fpol * mask + (1.0 - mask)*self.fvac()

        return fpol / R

    def psi(self):
        """
        Total poloidal flux ψ (psi), including contribution from
        plasma and external coils.
        """
        #return self.plasma_psi + self.tokamak.psi(self.R, self.Z)
        return self.plasma_psi + self.tokamak.calcPsiFromGreens(self._pgreen)
        
    def psiRZ(self, R, Z):
        """
        Return poloidal flux psi at given (R,Z) location
        """
        return self.psi_func(R, Z, grid=False) + self.tokamak.psi(R,Z)

    def fpol(self, psinorm):
        """
        Return f = R*Bt at specified values of normalised psi
        """
        return self._profiles.fpol(psinorm)
    
    def fvac(self):
        """
        Return vacuum f = R*Bt
        """
        return self._profiles.fvac()
    
    def q(self, psinorm = None, npsi=100):
        """
        Returns safety factor q at specified values of normalised psi
        
        psinorm is a scalar, list or array of floats betweem 0 and 1.
        
        >>> safety_factor = eq.q([0.2, 0.5, 0.9])
        
        If psinorm is None, then q on a uniform psi grid will be returned,
        along with the psi values

        >>> psinorm, q = eq.q()
        
        Note: psinorm = 0 is the magnetic axis, and psinorm = 1 is the separatrix.
              Calculating q on either of these flux surfaces is problematic,
              and the results will probably not be accurate.
        """
        if psinorm is None:
            # An array which doesn't include psinorm = 0 or 1
            psinorm = linspace(1./(npsi+1), 1.0, npsi, endpoint=False)
            return psinorm, critical.find_safety(self, psinorm=psinorm)

        result = critical.find_safety(self, psinorm=psinorm)
        # Convert to a scalar if only one result
        if len(result) == 1:
            return np.asscalar(result)
        return result
        
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
        return array(critical.find_separatrix(self, ntheta=ntheta, psi=self.psi()))[:, 0:2]

    def solve(self, profiles, Jtor=None, psi=None, psi_bndry=None):
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

        Jtor : 2D array
            If supplied, specifies the toroidal current at each (R,Z) point
            If not supplied, Jtor is calculated from profiles by finding O,X-points

        psi_bndry  - Poloidal flux to use as the separatrix (plasma boundary)
                     If not given then X-point locations are used.
        """
        
        self._profiles = profiles

        if Jtor is None:
            # Calculate toroidal current density

            if psi is None:
                psi = self.psi()
            Jtor = profiles.Jtor(self.R, self.Z, psi, psi_bndry=psi_bndry)
        
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
        Sets the plasma psi data, updates spline interpolation coefficients.
        Also updates:

        self.mask        2D (R,Z) array which is 1 in the core, 0 outside
        self.psi_axis    Value of psi on the magnetic axis
        self.psi_bndry   Value of psi on plasma boundary
        """
        self.plasma_psi = plasma_psi

        # Update spline interpolation
        self.psi_func = interpolate.RectBivariateSpline(self.R[:,0], self.Z[0,:], plasma_psi)

        # Update the locations of the X-points, core mask, psi ranges.
        # Note that this may fail if there are no X-points, so it should not raise an error
        # Analyse the equilibrium, finding O- and X-points
        psi = self.psi()
        opt, xpt = critical.find_critical(self.R, self.Z, psi)
        if opt:
            self.psi_axis = opt[0][2]

            if xpt:
                self.psi_bndry = xpt[0][2]
                self.mask = critical.core_mask(self.R, self.Z, psi, opt, xpt)
                
                # Use interpolation to find if a point is in the core.
                self.mask_func = interpolate.RectBivariateSpline(self.R[:,0], self.Z[0,:], self.mask)
            else:
                self.psi_bndry = None
                self.mask = None        

    def plot(self, axis=None, show=True, oxpoints=True):
        """
        Plot the equilibrium flux surfaces
        
        axis     - Specify the axis on which to plot
        show     - Call matplotlib.pyplot.show() before returning
        oxpoints - Plot X points as red circles, O points as green circles
    
        Returns
        -------

        axis  object from Matplotlib

        """
        from .plotting import plotEquilibrium
        return plotEquilibrium(self, axis=axis, show=show, oxpoints=oxpoints)
    
    def getForces(self):
        """
        Calculate forces on the coils

        Returns a dictionary of coil label -> force
        """
        return self.tokamak.getForces(self)

    def printForces(self):
        """
        Prints a table of forces on coils
        """
        print("Forces on coils")
        def print_forces(forces, prefix=""):
            for label, force in forces.items():
                if isinstance(force, dict):
                    print(prefix + label + " (circuit)")
                    print_forces(force, prefix=prefix + "  ")
                else:
                    print(prefix + label+ " : R = {0:.2f} kN , Z = {1:.2f} kN".format(force[0]*1e-3, force[1]*1e-3))

        print_forces(self.getForces())

    def innerOuterSeparatrix(self, Z = 0.0):
        """
        Locate R co ordinates of separatrix at both
        inboard and outboard poloidal midplane (Z = 0)
        """
        # Find the closest index to requested Z
        Zindex = np.argmin(abs(self.Z[0,:] - Z))

        # Normalise psi at this Z index
        psinorm = (self.psi()[:,Zindex] - self.psi_axis)  / (self.psi_bndry - self.psi_axis)
        
        # Start from the magnetic axis
        Rindex_axis = np.argmin(abs(self.R[:,0] - self.Rmagnetic()))

        # Inner separatrix
        # Get the maximum index where psi > 1 in the R index range from 0 to Rindex_axis
        outside_inds = np.argwhere(psinorm[:Rindex_axis] > 1.0)

        if outside_inds.size == 0:
            R_sep_in = self.Rmin
        else:
            Rindex_inner = np.amax(outside_inds)

            # Separatrix should now be between Rindex_inner and Rindex_inner+1
            # Linear interpolation
            R_sep_in = ((self.R[Rindex_inner, Zindex] * (1.0 - psinorm[Rindex_inner+1]) + 
                         self.R[Rindex_inner+1, Zindex] * (psinorm[Rindex_inner] - 1.0)) /
                        (psinorm[Rindex_inner] - psinorm[Rindex_inner+1]))

        # Outer separatrix
        # Find the minimum index where psi > 1
        outside_inds = np.argwhere(psinorm[Rindex_axis:] > 1.0)

        if outside_inds.size == 0:
            R_sep_out = self.Rmax
        else:
            Rindex_outer = np.amin(outside_inds) + Rindex_axis

            # Separatrix should now be between Rindex_outer-1 and Rindex_outer
            R_sep_out = ((self.R[Rindex_outer, Zindex] * (1.0 - psinorm[Rindex_outer-1]) + 
                          self.R[Rindex_outer-1, Zindex] * (psinorm[Rindex_outer] - 1.0)) /
                         (psinorm[Rindex_outer] - psinorm[Rindex_outer-1]))
        
        return R_sep_in, R_sep_out

    def intersectsWall(self):
        """Assess whether or not the core plasma touches the vessel
        walls. Returns True if it does intersect.
        """
        separatrix = self.separatrix() # Array [:,2]
        wall = self.tokamak.wall # Wall object with R and Z members (lists)
        
        return polygons.intersect(separatrix[:,0], separatrix[:,1],
                                  wall.R, wall.Z)
        
    def magneticAxis(self):
        """Returns the location of the magnetic axis as a list [R,Z,psi]
        """
        opt, xpt = critical.find_critical(self.R, self.Z, self.psi())
        return opt[0]

    def Rmagnetic(self):
        """The major radius R of magnetic major radius
        """
        return self.magneticAxis()[0]

    def geometricAxis(self, npoints=20):
        """Locates geometric axis, returning [R,Z]. Calculated as the centre
        of a large number of points on the separatrix equally
        distributed in angle from the magnetic axis.
        """
        separatrix = self.separatrix(ntheta=npoints) # Array [:,2]
        return np.mean(separatrix, axis=0)

    def Rgeometric(self, npoints=20):
        """Locates major radius R of the geometric major radius. Calculated
        as the centre of a large number of points on the separatrix
        equally distributed in angle from the magnetic axis.
        """
        return self.geometricAxis(npoints=npoints)[0]
    
    def minorRadius(self, npoints=20):
        """Calculates minor radius of plasma as the average distance from the
        geometric major radius to a number of points along the
        separatrix
        """
        separatrix = self.separatrix(ntheta=npoints)  # [:,2] 
        axis = np.mean(separatrix, axis=0) # Geometric axis [R,Z]

        # Calculate average distance from the geometric axis
        return np.mean( np.sqrt( (separatrix[:,0] - axis[0])**2 +   # dR^2
                                 (separatrix[:,1] - axis[1])**2 ))  # dZ^2

    def geometricElongation(self, npoints=20):
        """Calculates the elongation of a plasma using the range of R and Z of
        the separatrix

        """
        separatrix = self.separatrix(ntheta=npoints)  # [:,2]
        # Range in Z / range in R
        return (max(separatrix[:,1]) - min(separatrix[:,1])) / (max(separatrix[:,0]) - min(separatrix[:,0]))

    def aspectRatio(self, npoints=20):
        """Calculates the plasma aspect ratio

        """
        return self.Rgeometric(npoints=npoints)/ self.minorRadius(npoints=npoints)

    def effectiveElongation(self, R_wall_inner, R_wall_outer, npoints=300):
        """Calculates plasma effective elongation using the plasma volume

        """
        return self.plasmaVolume()/(2.*np.pi * self.Rgeometric(npoints=npoints) * self.minorRadius(npoints=npoints)**2)
    
    def internalInductance1(self, npoints=300):
        """Calculates li1 plasma internal inductance

        """
        # Produce array of Bpol^2 in (R,Z)
        B_polvals_2 = self.Bz(self.R,self.Z)**2 + self.Br(R,Z)**2

        R = self.R
        Z = self.Z
        dR = R[1,0] - R[0,0]
        dZ = Z[0,1] - Z[0,0]
        dV = 2. * np.pi * R * dR * dZ

        if self.mask is not None:   # Only include points in the core
            dV *= self.mask

        Ip = self.plasmaCurrent()
        R_geo = self.Rgeometric(npoints=npoints)
        elon = self.geometricElongation(npoints=npoints)
        effective_elon = self.effectiveElongation(npoints=npoints)
    
        integral = romb(romb(B_polvals_2*dV))
        return ((2 * integral) / ((mu0*Ip)**2 * R_geo))*( (1+elon*elon)/(2.*effective_elon) )

    def internalInductance2(self):
        """Calculates li2 plasma internal inductance

        """

        R = self.R
        Z = self.Z
        psi = self.psi()

        # Produce array of Bpol in (R,Z)
        B_polvals_2 = self.Br(R,Z)**2 + self.Bz(R,Z)**2

        dR = R[1,0] - R[0,0]
        dZ = Z[0,1] - Z[0,0]
        dV = 2.*np.pi*R * dR * dZ
        if self.mask is not None:   # Only include points in the core
            dV *= self.mask

        Ip = self.plasmaCurrent()
        R_mag = self.Rmagnetic()
    
        integral = romb(romb(B_polvals_2*dV))
        return 2 * integral / ((mu0*Ip)**2 * R_mag)

    def internalInductance3(self, R_wall_inner, R_wall_outer, npoints=300):
        """Calculates li3 plasma internal inductance

        """

        R = self.R
        Z = self.Z
        psi = self.psi()

        # Produce array of Bpol in (R,Z)
        B_polvals_2 = self.Br(R,Z)**2 + self.Bz(R,Z)**2

        dR = R[1,0] - R[0,0]
        dZ = Z[0,1] - Z[0,0]
        dV = 2.*np.pi*R * dR * dZ

        if self.mask is not None:   # Only include points in the core
            dV *= self.mask

        Ip = self.plasmaCurrent()
        R_geo = self.Rgeometric(npoints=npoints)
    
        integral = romb(romb(B_polvals_2*dV))
        return 2 * integral / ((mu0*Ip)**2 * R_geo)

    def poloidalBeta2(self):
        """Calculate plasma poloidal beta by integrating the thermal pressure
        and poloidal magnetic field pressure over the plasma volume.

        """

        R = self.R
        Z = self.Z

        # Produce array of Bpol in (R,Z)
        B_polvals_2 = self.Br(R,Z)**2 + self.Bz(R,Z)**2

        dR = R[1,0] - R[0,0]
        dZ = Z[0,1] - Z[0,0]
        dV = 2.*np.pi * R * dR * dZ

        # Normalised psi
        psi_norm = (self.psi() - self.psi_axis)  / (self.psi_bndry - self.psi_axis)
        
        # Plasma pressure
        pressure = self.pressure(psi_norm)

        if self.mask is not None: # Only include points in the core
            dV *= self.mask

        pressure_integral = romb(romb(pressure * dV))
        field_integral_pol = romb(romb(B_polvals_2 * dV))
        return 2 * mu0 * pressure_integral / field_integral_pol
        
        return poloidal_beta

    def toroidalBeta(self):
        """Calculate plasma toroidal beta by integrating the thermal pressure
        and toroidal magnetic field pressure over the plasma volume.

        """

        R = self.R
        Z = self.Z

        # Produce array of Btor in (R,Z)
        B_torvals_2 = self.Btor(R, Z)**2

        dR = R[1,0] - R[0,0]
        dZ = Z[0,1] - Z[0,0]
        dV = 2.*np.pi * R * dR * dZ
        
        # Normalised psi
        psi_norm = (self.psi() - self.psi_axis)  / (self.psi_bndry - self.psi_axis)
        
        # Plasma pressure
        pressure = self.pressure(psi_norm)

        if self.mask is not None: # Only include points in the core
            dV *= self.mask

        pressure_integral = romb(romb(pressure * dV))
        
        # Correct for errors in Btor and core masking
        np.nan_to_num(B_torvals_2, copy=False)

        field_integral_tor = romb(romb(B_torvals_2 * dV))
        return 2 * mu0 * pressure_integral / field_integral_tor

    def totalBeta(self):
        """Calculate plasma total beta

        """
        return 1./((1./self.poloidalBeta2()) + (1./self.toroidalBeta()))

def refine(eq, nx=None, ny=None):
    """
    Double grid resolution, returning a new equilibrium

    
    """
    # Interpolate the plasma psi
    #plasma_psi = multigrid.interpolate(eq.plasma_psi)
    #nx, ny = plasma_psi.shape
    
    # By default double the number of intervals
    if not nx:
        nx = 2*(eq.R.shape[0] - 1) + 1
    if not ny:
        ny = 2*(eq.R.shape[1] - 1) + 1

    result = Equilibrium(tokamak=eq.tokamak,
                         Rmin = eq.Rmin,
                         Rmax = eq.Rmax,
                         Zmin = eq.Zmin,
                         Zmax = eq.Zmax,
                         boundary=eq._applyBoundary,
                         order = eq.order,
                         nx=nx, ny=ny)
    
    plasma_psi = eq.psi_func(result.R, result.Z, grid=False)
    
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
