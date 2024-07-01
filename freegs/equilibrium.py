"""
Defines class to represent the equilibrium
state, including plasma and coil currents
"""

from numpy import pi, meshgrid, linspace, exp, array
import numpy as np
from scipy import interpolate
from scipy.integrate import romb, cumulative_trapezoid

from .boundary import fixedBoundary, freeBoundary
from . import critical
from . import polygons

# Operators which define the G-S equation
from .gradshafranov import mu0, GSsparse, GSsparse4thOrder

# Multigrid solver
from . import multigrid
from . import machine
import matplotlib.pyplot as plt

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

    def __init__(
        self,
        tokamak=machine.EmptyTokamak(),
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-1.0,
        Zmax=1.0,
        nx=65,
        ny=65,
        boundary=freeBoundary,
        psi=None,
        current=0.0,
        order=4,
        check_limited=False,
    ):
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

        check_limited - Boolean, checks if the plasma is limited.
        """

        self.tokamak = tokamak

        self._applyBoundary = boundary

        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Zmin = Zmin
        self.Zmax = Zmax
        self.nx = nx
        self.ny = ny

        self.R_1D = linspace(Rmin, Rmax, nx)
        self.Z_1D = linspace(Zmin, Zmax, ny)
        self.R, self.Z = meshgrid(self.R_1D, self.Z_1D, indexing="ij")

        self.dR = self.R[1, 0] - self.R[0, 0]
        self.dZ = self.Z[0, 1] - self.Z[0, 0]

        self.check_limited = check_limited
        self.is_limited = False
        self.Rlim = None
        self.Zlim = None

        if psi is None:
            # Starting guess for psi
            xx, yy = meshgrid(linspace(0, 1, nx), linspace(0, 1, ny), indexing="ij")
            psi = exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.4 ** 2)

            psi[0, :] = 0.0
            psi[:, 0] = 0.0
            psi[-1, :] = 0.0
            psi[:, -1] = 0.0

        # Calculate coil Greens functions. This is an optimisation,
        # used in self.psi() to speed up calculations
        self._pgreen = tokamak.createPsiGreens(self.R, self.Z)

        self._current = current  # Plasma current
        self.Jtor = None
        self._updatePlasmaPsi(psi)  # Needs to be after _pgreen

        # Create the solver
        if order == 2:
            generator = GSsparse(Rmin, Rmax, Zmin, Zmax)
        elif order == 4:
            generator = GSsparse4thOrder(Rmin, Rmax, Zmin, Zmax)
        else:
            raise ValueError(
                "Invalid choice of order ({}). Valid values are 2 or 4.".format(order)
            )
        self.order = order

        self._solver = multigrid.createVcycle(
            nx, ny, generator, nlevels=1, ncycle=1, niter=2, direct=True
        )

    def setSolverVcycle(self, nlevels=1, ncycle=1, niter=1, direct=True):
        """
        Creates a new linear solver, based on the multigrid code

        nlevels  - Number of resolution levels, including original
        ncycle   - The number of V cycles
        niter    - Number of linear solver (Jacobi) iterations per level
        direct   - Use a direct solver at the coarsest level?

        """
        generator = GSsparse(self.Rmin, self.Rmax, self.Zmin, self.Zmax)
        nx, ny = self.R.shape

        self._solver = multigrid.createVcycle(
            nx,
            ny,
            generator,
            nlevels=nlevels,
            ncycle=ncycle,
            niter=niter,
            direct=direct,
        )

    def setSolver(self, solver):
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

    def plasmaVolume(self):
        """Calculate the volume of the plasma in m^3"""

        dR = self.R[1, 0] - self.R[0, 0]
        dZ = self.Z[0, 1] - self.Z[0, 0]

        # Volume element
        dV = 2.0 * pi * self.R * dR * dZ

        if self.mask is not None:  # Only include points in the core
            dV *= self.mask

        # Integrate volume in 2D
        return romb(romb(dV))

    def plasmaBr(self, R, Z):
        """
        Radial magnetic field due to plasma
        Br = -1/R dpsi/dZ
        """
        return -self.psi_func(R, Z, dy=1, grid=False) / R

    def plasmaBz(self, R, Z):
        """
        Vertical magnetic field due to plasma
        Bz = (1/R) dpsi/dR
        """
        return self.psi_func(R, Z, dx=1, grid=False) / R

    def Br(self, R, Z):
        """
        Total radial magnetic field
        """
        return self.plasmaBr(R, Z) + self.tokamak.Br(R, Z)

    def Bz(self, R, Z):
        """
        Total vertical magnetic field
        """
        return self.plasmaBz(R, Z) + self.tokamak.Bz(R, Z)

    def Bpol(self, R, Z):
        """
        Total poloidal magnetic field
        """
        Br = self.Br(R, Z)
        Bz = self.Bz(R, Z)
        return np.sqrt(Br * Br + Bz * Bz)

    def Btor(self, R, Z):
        """
        Toroidal magnetic field
        """
        # Normalised psi
        psi_norm = (self.psiRZ(R, Z) - self.psi_axis) / (self.psi_bndry - self.psi_axis)

        # Get f = R * Btor in the core. May be invalid outside the core
        fpol = self.fpol(psi_norm)

        if self.mask is not None:
            # Get the values of the core mask at the requested R,Z locations
            # This is 1 in the core, 0 outside
            mask = self.mask_func(R, Z, grid=False)
            fpol = fpol * mask + (1.0 - mask) * self.fvac()

        return fpol / R

    def Btot(self, R, Z):
        """
        Total magnetic field
        """
        Br = self.Br(R, Z)
        Bz = self.Bz(R, Z)
        Btor = self.Btor(R, Z)
        return np.sqrt(Br * Br + Bz * Bz + Btor * Btor)

    def psi(self):
        """
        Total poloidal flux (psi), including contribution from
        plasma and external coils.
        """
        # return self.plasma_psi + self.tokamak.psi(self.R, self.Z)
        return self.plasma_psi + self.tokamak.calcPsiFromGreens(self._pgreen)

    def psiN(self):
        """
        Total poloidal flux (psi), including contribution from
        plasma and external coils. Normalised such that psiN = 0 on
        the magnetic axis and 1 on the LCFS.
        """
        # return self.plasma_psi + self.tokamak.psi(self.R, self.Z)
        return (self.psi() - self.psi_axis) / (self.psi_bndry - self.psi_axis)

    def psiRZ(self, R, Z):
        """
        Return poloidal flux psi at given (R,Z) location
        """
        return self.psi_func(R, Z, grid=False) + self.tokamak.psi(R, Z)

    def psiNRZ(self, R, Z):
        """
        Return poloidal flux psi at given (R,Z) location. Normalised such
        that psiN = 0 on the magnetic axis and 1 on the LCFS.
        """
        return (self.psiRZ(R, Z) - self.psi_axis) / (self.psi_bndry - self.psi_axis)

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

    def q(self, psinorm=None, npsi=100):
        """
        Returns safety factor q at specified values of normalised psi

        psinorm is a scalar, list or array of floats betweem 0 and 1.

        >>> safety_factor = eq.q([0.2, 0.5, 0.9])

        If psinorm is None, then q on a uniform psi grid will be returned,
        along with the psi values

        >>> psinorm, q = eq.q()

        Note: psinorm = 0 is the magnetic axis, and psinorm = 1 is the separatrix.
              If you request a value close to either of these limits, an extrapolation
              based on a 1D grid of values from 0.01 to 0.99 will be used. This gives
              smooth and continuous q-profiles, but comes at an increased computational
              cost.
        """
        if psinorm is None:
            # An array which doesn't include psinorm = 0 or 1
            psinorm = linspace(1.0 / (npsi + 1), 1.0, npsi, endpoint=False)
            return psinorm, critical.find_safety(self, psinorm=psinorm)

        elif np.any((psinorm < 0.01) | (psinorm > 0.99)):
            psinorm_inner = np.linspace(0.01, 0.99, num=npsi)
            q_inner = critical.find_safety(self, psinorm=psinorm_inner)

            interp = interpolate.interp1d(psinorm_inner, q_inner,
                                          kind = "quadratic",
                                          fill_value = "extrapolate")
            result = interp(psinorm)
        else:
            result = critical.find_safety(self, psinorm=psinorm)
        
        # Convert to a scalar if only one result
        if np.size(result) == 1:
            return float(result)
        return result

    def tor_flux(self, psi=None):
        """
        Calculates toroidal flux at specified values of poloidal flux.
        >>> q = drho/dpsi
        """
        psiN = (psi - self.psi_axis) / (self.psi_bndry - self.psi_axis)
        # Get safety factor of these flux surfaces
        qvals = self.q(psiN)

        # Integrate q wrt psi to get rho. rho = 0 @ psiN = 0
        result = cumulative_trapezoid(qvals, psi, initial=0.0) * (-1.0 / (2.0 * np.pi))

        # Convert to a scalar if only one result
        if len(result) == 1:
            return result[0]
        return result

    def rhotor(self, psi=None):
        """
        Calculates normalised toroidal flux at specified values of
        poloidal flux.
         >>> rhotor = sqrt ( tor_flux/max(tor_flux)).

        Maximum toroidal flux shoud be at LCFS.
        """

        torflux = self.tor_flux(psi)

        psi = np.linspace(self.psi_axis, self.psi_bndry, 101, endpoint=True)
        torflux_for_LCFS = self.tor_flux(psi)

        max_torflux = np.max(torflux_for_LCFS)

        result = np.sqrt(torflux / max_torflux)

        if len(result) == 1:
            return result[0]
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

    def pressure(self, psinorm):
        """
        Returns plasma pressure at specified values of normalised psi
        """
        return self._profiles.pressure(psinorm)

    def separatrix(self, npoints=360):
        """
        Returns an array of npoints (R, Z) coordinates of the separatrix,
        equally spaced in geometric poloidal angle.
        """
        return array(critical.find_separatrix(self, ntheta=npoints, psi=self.psi()))[
            :, 0:2
        ]

    def psi_surfRZ(self, psiN=0.995, npoints=360):
        """
        Returns the R,Z of a flux surface specified by a value of psiN. This flux surface is closed on itself.
        """
        
        surf = critical.find_separatrix(self, opoint=None, xpoint=None, ntheta=npoints, psi=None, axis=None, psival=psiN)

        Rsurf = [point[0] for point in surf]
        Zsurf = [point[1] for point in surf]

        Rsurf.append(Rsurf[0])
        Zsurf.append(Zsurf[0])

        return np.array(Rsurf), np.array(Zsurf)

    def solve(self, profiles, Jtor=None, psi=None, psi_bndry=None):
        """
        Calculate the plasma equilibrium given new profiles
        replacing the current equilibrium.

        This performs the linear Grad-Shafranov solve

        profiles  - An object describing the plasma profiles.
                    At minimum this must have methods:
             .Jtor(R, Z, psi, psi_bndry)   -> [nx, ny]
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
        self._updateBoundaryPsi()

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
        rhs[0, :] = self.plasma_psi[0, :]
        rhs[:, 0] = self.plasma_psi[:, 0]
        rhs[-1, :] = self.plasma_psi[-1, :]
        rhs[:, -1] = self.plasma_psi[:, -1]

        # Call elliptic solver
        plasma_psi = self._solver(self.plasma_psi, rhs)

        self._updatePlasmaPsi(plasma_psi)

        # Update plasma current
        dR = self.R[1, 0] - self.R[0, 0]
        dZ = self.Z[0, 1] - self.Z[0, 0]
        self._current = romb(romb(Jtor)) * dR * dZ
        self.Jtor = Jtor

    def _updateBoundaryPsi(self, psi=None):
        """
        For an input psi the magnetic axis and boundary psi are identified along
        with the core mask.

        Various logical checks occur, depending on whether or not the user
        wishes to check if the plasma is limited or not, as well as whether
        or not any xpoints are present.
        """

        if psi is None:
            psi = self.psi()

        opt, xpt = critical.find_critical(self.R, self.Z, psi)

        psi = psi

        if opt:
            # Magnetic axis flux taken as primary o-point flux
            self.psi_axis = opt[0][2]
            """
            Several options depending on if user wishes to check
            if the plasma becomes limited.
            """

            # The user wishes to check if the plasma is limited
            if self.check_limited and self.tokamak.wall:
                # A wall has actually been provided, proceed with checking

                # Obtain flux on machine limit points
                Rlimit = self.tokamak.limit_points_R
                Zlimit = self.tokamak.limit_points_Z

                """
                If an xpoint is present (plasma is potentianlly diverted)
                then we must remove any limit points above/below the
                primary xpoint as the PFR may land on these points,
                which would break the algorithm (at present) for extracting the boundary
                flux if the plasma were to infact be limited. There is a more advanced
                version of this alogorithm that is more robust that will be
                added in the future.
                """

                if xpt:
                    limit_args = np.ravel(
                        np.argwhere(abs(Zlimit) < abs(0.75 * xpt[0][1]))
                    )
                    Rlimit = Rlimit[limit_args]
                    Zlimit = Zlimit[limit_args]

                # Obtain the flux psi at these limiter points
                R = np.asarray(self.R[:, 0])
                Z = np.asarray(self.Z[0, :])

                # psi is transposed due to how FreeGS meshgrids R,Z
                psi_2d = interpolate.interp2d(x=R, y=Z, z=psi.T)

                # Get psi at the limit points
                psi_limit_points = np.zeros(len(Rlimit))
                for i in range(len(Rlimit)):
                    psi_limit_points[i] = psi_2d(Rlimit[i], Zlimit[i])[0]

                # Get index of maximum psi value
                indMax = np.argmax(psi_limit_points)

                # Extract R,Z of the contact point
                self.Rlim = Rlimit[indMax]
                self.Zlim = Zlimit[indMax]

                # Obtain maximum psi
                self.psi_limit = psi_limit_points[indMax]

                # Check if any xpoints are present
                if xpt:
                    # Get flux from the primary xpoint
                    self.psi_xpt = xpt[0][2]

                    # Choose between diverted or limited flux
                    self.psi_bndry = max(self.psi_xpt, self.psi_limit)

                    if self.psi_bndry == self.psi_limit:
                        self.is_limited = True

                    else:
                        self.is_limited = False

                    # Mask the core
                    self.mask = critical.core_mask(
                        self.R, self.Z, psi, opt, xpt, self.psi_bndry
                    )

                    # Use interpolation to find if a point is in the core.
                    self.mask_func = interpolate.RectBivariateSpline(
                        self.R[:, 0], self.Z[0, :], self.mask
                    )

                else:
                    # No xpoints, therefore psi_bndry = psi_limit
                    self.psi_bndry = self.psi_limit
                    self.is_limited = True
                    self.mask = None

            else:
                # Either a wall was not provided or the user did not wish to
                # check if the plasma was limited
                if xpt:
                    self.psi_xpt = xpt[0][2]
                    self.psi_bndry = self.psi_xpt
                    self.mask = critical.core_mask(self.R, self.Z, psi, opt, xpt)

                    # Use interpolation to find if a point is in the core.
                    self.mask_func = interpolate.RectBivariateSpline(
                        self.R[:, 0], self.Z[0, :], self.mask
                    )
                elif self._applyBoundary is fixedBoundary:
                    # No X-points, but using fixed boundary
                    self.psi_bndry = psi[0, 0]  # Value of psi on the boundary
                    self.mask = None  # All points are in the core region
                else:
                    self.psi_bndry = None
                    self.mask = None

                self.is_limited = False

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
        self.psi_func = interpolate.RectBivariateSpline(
            self.R[:, 0], self.Z[0, :], plasma_psi
        )
        
        # Update the plasma axis and boundary flux as well as mask
        self._updateBoundaryPsi()

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
                    print(
                        prefix
                        + label
                        + " : R = {0:.2f} kN , Z = {1:.2f} kN".format(
                            force[0] * 1e-3, force[1] * 1e-3
                        )
                    )

        print_forces(self.getForces())

    def innerOuterSeparatrix(self, Z=0.0):
        """
        Locate R co ordinates of separatrix at both
        inboard and outboard poloidal midplane (Z = 0)
        """
        # Find the closest index to requested Z
        Zindex = np.argmin(abs(self.Z[0, :] - Z))

        # Normalise psi at this Z index
        psinorm = (self.psi()[:, Zindex] - self.psi_axis) / (
            self.psi_bndry - self.psi_axis
        )

        # Start from the magnetic axis
        Rindex_axis = np.argmin(abs(self.R[:, 0] - self.Rmagnetic()))

        # Inner separatrix
        # Get the maximum index where psi > 1 in the R index range from 0 to Rindex_axis
        outside_inds = np.argwhere(psinorm[:Rindex_axis] > 1.0)

        if outside_inds.size == 0:
            R_sep_in = self.Rmin
        else:
            Rindex_inner = np.amax(outside_inds)

            # Separatrix should now be between Rindex_inner and Rindex_inner+1
            # Linear interpolation
            R_sep_in = (
                self.R[Rindex_inner, Zindex] * (1.0 - psinorm[Rindex_inner + 1])
                + self.R[Rindex_inner + 1, Zindex] * (psinorm[Rindex_inner] - 1.0)
            ) / (psinorm[Rindex_inner] - psinorm[Rindex_inner + 1])

        # Outer separatrix
        # Find the minimum index where psi > 1
        outside_inds = np.argwhere(psinorm[Rindex_axis:] > 1.0)

        if outside_inds.size == 0:
            R_sep_out = self.Rmax
        else:
            Rindex_outer = np.amin(outside_inds) + Rindex_axis

            # Separatrix should now be between Rindex_outer-1 and Rindex_outer
            R_sep_out = (
                self.R[Rindex_outer, Zindex] * (1.0 - psinorm[Rindex_outer - 1])
                + self.R[Rindex_outer - 1, Zindex] * (psinorm[Rindex_outer] - 1.0)
            ) / (psinorm[Rindex_outer] - psinorm[Rindex_outer - 1])

        return R_sep_in, R_sep_out

    def intersectsWall(self):
        """Assess whether or not the core plasma touches the vessel
        walls. Returns True if it does intersect.
        """
        separatrix = self.separatrix()  # Array [:,2]
        wall = self.tokamak.wall  # Wall object with R and Z members (lists)

        return polygons.intersect(separatrix[:, 0], separatrix[:, 1], wall.R, wall.Z)

    def magneticAxis(self):
        """Returns the location of the magnetic axis as a list [R,Z,psi]"""
        opt, xpt = critical.find_critical(self.R, self.Z, self.psi())
        return opt[0]

    def Rmagnetic(self):
        """The major radius R of magnetic major radius"""
        return self.magneticAxis()[0]

    def Zmagnetic(self):
        """The height Z of magnetic axis"""
        return self.magneticAxis()[1]

    def geometricAxis(self, npoints=360):
        """Locates geometric axis, returning [R,Z]. First locates the
        extrema points in R of the LCFS, wherein P3 is at the IMP and
        P1 is at the OMP.

        R0 = R(P3) + 0.5*(R(P1)-R(P3))
        z0 = 0.5*(Z(P1)+Z(P3))
        """

        # Get points along the LCFS
        separatrix = self.separatrix(npoints=npoints)  # Array [:,2]

        Rlcfs = np.array([i[0] for i in separatrix])
        Zlcfs = np.array([i[1] for i in separatrix])

        ind_P1 = np.argmax(Rlcfs)
        ind_P3 = np.argmin(Rlcfs)

        P1 = np.array([Rlcfs[ind_P1], Zlcfs[ind_P1]])
        P3 = np.array([Rlcfs[ind_P3], Zlcfs[ind_P3]])

        R0 = P3[0] + 0.5 * (P1[0] - P3[0])
        z0 = 0.5 * (P1[1] + P3[1])

        C = np.array([R0, z0])

        return C

    def Rgeometric(self, npoints=360):
        """Locates major radius R of the geometric major radius."""
        return self.geometricAxis(npoints=npoints)[0]

    def Zgeometric(self, npoints=360):
        """Locates the height z of the geometric axis."""
        return self.geometricAxis(npoints=npoints)[1]

    def minorRadius(self, npoints=360):
        """Calculates minor radius of the plasma, a. First locates the
        extrema points in R of the LCFS, wherein P3 is at the IMP and
        P1 is at the OMP.

        a = 0.5*(R(P1) - R(P3))
        """

        # Get points along the LCFS
        separatrix = self.separatrix(npoints=npoints)  # Array [:,2]

        Rlcfs = np.array([i[0] for i in separatrix])

        R_P1 = np.max(Rlcfs)
        R_P3 = np.min(Rlcfs)

        return 0.5 * (R_P1 - R_P3)

    def aspectRatio(self, npoints=360):
        """Calculates the plasma aspect ratio.

        A = R0/a where R0 = major radius, a = minor radius.
        """
        return self.Rgeometric(npoints=npoints) / self.minorRadius(npoints=npoints)

    def inverseAspectRatio(self, npoints=360):
        """Calculates inverse of the plasma aspect ratio.

        epsilon = 1/A
        A = R0/a where R0 = major radius, a = minor radius.
        """
        return self.minorRadius(npoints=npoints) / self.Rgeometric(npoints=npoints)

    def elongation(self, npoints=360):
        """Calculates the elongation, kappa, of the plasma. A large number
        of points should be supplied such that any primary xpoint(s) on
        the LCFS are captured. The R,Z of the primary x-point is NOT
        itself included in the R,Z of the LCFS as the plasma may be limited.
        P2 is the point at the upper extent of the plasma, and P4 is the point
        at the lower extent of the plasma.

        kappa = (Z(P2) - Z(P4))/a
        """

        # Get points along the LCFS
        separatrix = self.separatrix(npoints=npoints)  # Array [:,2]

        Zlcfs = np.array([i[1] for i in separatrix])

        Z_P2 = np.max(Zlcfs)
        Z_P4 = np.min(Zlcfs)

        a = self.minorRadius(npoints=npoints)

        return 0.5 * (Z_P2 - Z_P4) / a

    def elongationUpper(self, npoints=360):
        """Calculates the upper elongation, kappa_u, of the plasma. A large number
        of points should be supplied such that any primary xpoint(s) on
        the LCFS are captured. The R,Z of the primary x-point is NOT
        itself included in the R,Z of the LCFS as the plasma may be limited.
        P2 is the point at the upper extent of the plasma.

        kappa_u = (Z(P2) - z0)/a
        """

        # Get points along the LCFS
        separatrix = self.separatrix(npoints=npoints)  # Array [:,2]

        Zlcfs = np.array([i[1] for i in separatrix])

        Z_P2 = np.max(Zlcfs)

        z0 = self.Zgeometric(npoints=npoints)
        a = self.minorRadius(npoints=npoints)

        return (Z_P2 - z0) / a

    def elongationLower(self, npoints=360):
        """Calculates the lower elongation, kappa_l, of the plasma. A large number
        of points should be supplied such that any primary xpoint(s) on
        the LCFS are captured. The R,Z of the primary x-point is NOT
        itself included in the R,Z of the LCFS as the plasma may be limited.
        P2 is the point at the upper extent of the plasma.

        kappa_u = (z0 - Z(P4))/a
        """

        # Get points along the LCFS
        separatrix = self.separatrix(npoints=npoints)  # Array [:,2]

        Zlcfs = np.array([i[1] for i in separatrix])

        Z_P4 = np.min(Zlcfs)

        z0 = self.Zgeometric(npoints=npoints)
        a = self.minorRadius(npoints=npoints)

        return (z0 - Z_P4) / a

    def effectiveElongation(self, npoints=360):
        """Calculates plasma effective elongation using the plasma volume"""
        return self.plasmaVolume() / (
            2.0
            * np.pi
            * self.Rgeometric(npoints=npoints)
            * np.pi
            * self.minorRadius(npoints=npoints) ** 2
        )

    def triangularityUpper(self, npoints=360):
        """Calculates plasma upper triangularity, delta_u.
        P2 is the point at the upper extent of the plasma.

        tri_u = (R0 - R(P2))/a
        """

        # Get points along the LCFS
        separatrix = self.separatrix(npoints=npoints)  # Array [:,2]
        Rlcfs = np.array([i[0] for i in separatrix])
        Zlcfs = np.array([i[1] for i in separatrix])

        ind_P2 = np.argmax(Zlcfs)

        R_P2 = Rlcfs[ind_P2]

        R0 = self.Rgeometric(npoints=npoints)
        a = self.minorRadius(npoints=npoints)

        return (R0 - R_P2) / a

    def triangularityLower(self, npoints=360):
        """Calculates plasma upper triangularity, delta_u.
        P4 is the point at the lower extent of the plasma.

        tri_l = (R0 - R(P4))/a
        """

        # Get points along the LCFS
        separatrix = self.separatrix(npoints=npoints)  # Array [:,2]
        Rlcfs = np.array([i[0] for i in separatrix])
        Zlcfs = np.array([i[1] for i in separatrix])

        ind_P2 = np.argmax(Zlcfs)

        R_P2 = Rlcfs[ind_P2]

        R0 = self.Rgeometric(npoints=npoints)
        a = self.minorRadius(npoints=npoints)

        return (R0 - R_P2) / a

    def triangularity(self, npoints=360):
        """Calculates plasma triangularity, delta.

        Here delta is defined as the average of the upper
        and lower triangularities.
        """

        tri_u = self.triangularityUpper(npoints=npoints)
        tri_l = self.triangularityLower(npoints=npoints)

        return 0.5 * (tri_u + tri_l)

    def shafranovShift(self, npoints=360):
        """Calculates the plasma shafranov shift
        [delta_shafR,delta_shafZ] where

        delta_shafR = Rmagnetic - Rgeo
        delta_shafR = Zmagnetic - z0
        """

        Rmag = self.Rmagnetic()
        Zmag = self.Zmagnetic()

        Rgeo = self.Rgeometric()
        z0 = self.Zgeometric()

        return np.array([Rmag - Rgeo, Zmag - z0])

    def internalInductance1(self, npoints=360):
        """Calculates li1 plasma internal inductance"""

        R = self.R
        Z = self.Z
        # Produce array of Bpol^2 in (R,Z)
        B_polvals_2 = self.Bz(R, Z) ** 2 + self.Br(R, Z) ** 2

        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]
        dV = 2.0 * np.pi * R * dR * dZ

        if self.mask is not None:  # Only include points in the core
            dV *= self.mask

        Ip = self.plasmaCurrent()
        R_geo = self.Rgeometric(npoints=npoints)
        elon = self.elongation(npoints=npoints)
        effective_elon = self.effectiveElongation(npoints=npoints)

        integral = romb(romb(B_polvals_2 * dV))
        return ((2 * integral) / ((mu0 * Ip) ** 2 * R_geo)) * (
            (1 + elon * elon) / (2.0 * effective_elon)
        )

    def internalInductance2(self):
        """Calculates li2 plasma internal inductance"""

        R = self.R
        Z = self.Z
        # Produce array of Bpol in (R,Z)
        B_polvals_2 = self.Br(R, Z) ** 2 + self.Bz(R, Z) ** 2

        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]
        dV = 2.0 * np.pi * R * dR * dZ
        if self.mask is not None:  # Only include points in the core
            dV *= self.mask

        Ip = self.plasmaCurrent()
        R_mag = self.Rmagnetic()

        integral = romb(romb(B_polvals_2 * dV))
        return 2 * integral / ((mu0 * Ip) ** 2 * R_mag)

    def internalInductance3(self, npoints=360):
        """Calculates li3 plasma internal inductance"""

        R = self.R
        Z = self.Z
        # Produce array of Bpol in (R,Z)
        B_polvals_2 = self.Br(R, Z) ** 2 + self.Bz(R, Z) ** 2

        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]
        dV = 2.0 * np.pi * R * dR * dZ

        if self.mask is not None:  # Only include points in the core
            dV *= self.mask

        Ip = self.plasmaCurrent()
        R_geo = self.Rgeometric(npoints=npoints)

        integral = romb(romb(B_polvals_2 * dV))
        return 2 * integral / ((mu0 * Ip) ** 2 * R_geo)

    def internalInductance(self, npoints=360):
        """Calculates plasma internal inductance li

        li = 4/(mu0*R0*Ip^2) * int(2piR*(Bp^2/2mu0)*dR*dZ)
           = 2/(mu0^2*R0*Ip^2)*int(2piR*Bp^2*dR*dZ)
        """

        R = self.R
        Z = self.Z
        # Produce array of Bpol in (R,Z)
        B_polvals_2 = self.Br(R, Z) ** 2 + self.Bz(R, Z) ** 2

        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]
        dV = 2.0 * np.pi * R * dR * dZ

        if self.mask is not None:  # Only include points in the core
            dV *= self.mask

        Ip = self.plasmaCurrent()
        R_geo = self.Rgeometric(npoints=npoints)

        integral = romb(romb(B_polvals_2 * dV))
        return 2 * integral / (mu0 * mu0 * R_geo * Ip * Ip)

    def flux_surface_averaged_Bpol2(self, psiN=0.995, npoints=360):
        """
        Calculates the flux surface averaged value of the square of the poloidal field.
        """

        # Get R, Z points of the flux surface
        Rsurf, Zsurf = self.psi_surfRZ(psiN=psiN,npoints=npoints)

        # Get the poloidal field
        Bpol_surf = self.Bpol(Rsurf,Zsurf)

        # Get the square of the poloidal field
        Bpol_surf2 = Bpol_surf**2.0

        # Get dl along the surface
        dl = np.sqrt(np.diff(Rsurf)**2.0 + np.diff(Zsurf)**2.0)
        dl = np.insert(dl,0,0.0)

        # Get l along the surface
        l = np.cumsum(dl)

        # Calculate the flux surface averaged quantity
        return np.trapz(x=l, y=Bpol_surf2 * Bpol_surf) / np.trapz(x=l, y=np.ones(np.size(l)) * Bpol_surf)

    def poloidalBeta(self):
        """Return the poloidal beta.
        
        betaP = 2 * mu0 * <p> / <<Bpol^2>>
        """

        # Normalised psi
        psi_norm = self.psiN()

        # Plasma pressure
        pressure = self.pressure(psi_norm)

        volume_averaged_pressure = self.calc_volume_averaged(pressure)
        line_averaged_Bpol2_lcfs = self.flux_surface_averaged_Bpol2(psiN=1.0)

        return (2.0 * mu0 * volume_averaged_pressure) / line_averaged_Bpol2_lcfs

    def poloidalBeta2(self):
        """Return the poloidal beta
        betap = (8pi/mu0) * int(p)dRdZ / Ip^2
        """

        dR = self.R[1, 0] - self.R[0, 0]
        dZ = self.Z[0, 1] - self.Z[0, 0]

        # Normalised psi
        psi_norm = (self.psi() - self.psi_axis) / (self.psi_bndry - self.psi_axis)

        # Plasma pressure
        pressure = self.pressure(psi_norm)
        if self.mask is not None:
            # If there is a masking function (X-points, limiters)
            pressure *= self.mask

        # Integrate pressure in 2D
        return (
            ((8.0 * pi) / mu0)
            * romb(romb(pressure))
            * dR
            * dZ
            / (self.plasmaCurrent() ** 2)
        )

    def poloidalBeta3(self):
        """Calculates alterantive poloidal beta definition."""

        R = self.R
        Z = self.Z

        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]
        dV = 2.0 * np.pi * R * dR * dZ

        # Normalised psi
        psi_norm = (self.psi() - self.psi_axis) / (self.psi_bndry - self.psi_axis)

        # Plasma pressure
        pressure = self.pressure(psi_norm)

        if self.mask is not None:  # Only include points in the core
            dV *= self.mask

        pressure_integral = romb(romb(pressure * dV))
        Ip = self.plasmaCurrent()
        vol = self.plasmaVolume()
        r0 = self.Rgeometric()
        return 4 * vol * pressure_integral / (mu0 * Ip * Ip * r0)

    def toroidalBeta(self):
        """Calculate plasma toroidal beta by integrating the thermal pressure
        and toroidal magnetic field pressure over the plasma volume."""

        R = self.R
        Z = self.Z

        # Produce array of Btor in (R,Z)
        B_torvals_2 = self.Btor(R, Z) ** 2

        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]
        dV = 2.0 * np.pi * R * dR * dZ

        # Normalised psi
        psi_norm = (self.psi() - self.psi_axis) / (self.psi_bndry - self.psi_axis)

        # Plasma pressure
        pressure = self.pressure(psi_norm)

        if self.mask is not None:  # Only include points in the core
            dV *= self.mask

        pressure_integral = romb(romb(pressure * dV))

        # Correct for errors in Btor and core masking
        np.nan_to_num(B_torvals_2, copy=False)

        field_integral_tor = romb(romb(B_torvals_2 * dV))
        return 2 * mu0 * pressure_integral / field_integral_tor

    def totalBeta(self):
        """Calculate plasma total beta"""
        return 1.0 / ((1.0 / self.poloidalBeta()) + (1.0 / self.toroidalBeta()))

    def betaN(self, npoints=360):
        """Calculate normalised plasma beta"""
        geo = self.geometricAxis()
        Bt = self.Btor(geo[0], geo[1])
        return (
            100.0
            * 1.0e06
            * self.toroidalBeta()
            * ((self.minorRadius() * Bt) / (self.plasmaCurrent()))
        )

    def pressure_ave(self):
        """Calculate average pressure, Pa."""

        R = self.R
        Z = self.Z

        # Produce array of Btor in (R,Z)
        B_torvals_2 = self.Btor(R, Z) ** 2

        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]
        dV = 2.0 * np.pi * R * dR * dZ

        # Normalised psi
        psi_norm = (self.psi() - self.psi_axis) / (self.psi_bndry - self.psi_axis)

        # Plasma pressure
        pressure = self.pressure(psi_norm)

        if self.mask is not None:  # Only include points in the core
            dV *= self.mask

        pressure_integral = romb(romb(pressure * dV))
        plasmaVolume = romb(romb(dV))

        return pressure_integral / plasmaVolume

    def w_th(self):
        """
        Stored thermal energy in plasma, J.
        """

        R = self.R
        Z = self.Z

        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]
        dV = 2.0 * np.pi * R * dR * dZ

        # Normalised psi
        psi_norm = (self.psi() - self.psi_axis) / (self.psi_bndry - self.psi_axis)

        # Plasma pressure
        pressure = self.pressure(psi_norm)

        if self.mask is not None:  # Only include points in the core
            dV *= self.mask

        pressure_integral = romb(romb(pressure * dV))
        thermal_energy = (3.0 / 2.0) * pressure_integral

        return thermal_energy

    def qcyl(self):
        """
        Cylindrical safety factor.
        """

        eps = self.inverseAspectRatio()
        a = self.minorRadius()

        btor = self.fvac() / self.Rgeometric()
        Ip = self.plasmaCurrent()

        kappa = self.elongation()

        val = 0.5 * (1 + kappa * kappa) * ((2.0 * np.pi * a * eps * btor) / (mu0 * Ip))

        return val

    def calc_volume_integrated(self,field):
        """
        Calculates the volume integral of the input field.
        """

        dV = 2.0 * np.pi * self.R * self.dR  *self.dZ

        if self.mask is not None:  # Only include points in the core
            dV *= self.mask

        return romb(romb(field * dV))

    def calc_volume_averaged(self,field):
        """
        Calculates the volume average of the input field.
        """

        volume_integrated_field = self.calc_volume_integrated(field)
        
        return volume_integrated_field / self.plasmaVolume()

def refine(eq, nx=None, ny=None):
    """
    Double grid resolution, returning a new equilibrium


    """
    # Interpolate the plasma psi
    # plasma_psi = multigrid.interpolate(eq.plasma_psi)
    # nx, ny = plasma_psi.shape

    # By default double the number of intervals
    if not nx:
        nx = 2 * (eq.R.shape[0] - 1) + 1
    if not ny:
        ny = 2 * (eq.R.shape[1] - 1) + 1

    result = Equilibrium(
        tokamak=eq.tokamak,
        Rmin=eq.Rmin,
        Rmax=eq.Rmax,
        Zmin=eq.Zmin,
        Zmax=eq.Zmax,
        boundary=eq._applyBoundary,
        order=eq.order,
        nx=nx,
        ny=ny,
    )

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

    result = Equilibrium(
        tokamak=eq.tokamak,
        Rmin=eq.Rmin,
        Rmax=eq.Rmax,
        Zmin=eq.Zmin,
        Zmax=eq.Zmax,
        nx=nx,
        ny=ny,
    )

    result._updatePlasmaPsi(plasma_psi)

    if hasattr(eq, "_profiles"):
        result._profiles = eq._profiles

    if hasattr(eq, "control"):
        result.control = eq.control

    return result

def newDomain(eq, Rmin=None, Rmax=None, Zmin=None, Zmax=None, nx=None, ny=None):
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
    result = Equilibrium(
        tokamak=eq.tokamak, Rmin=Rmin, Rmax=Rmax, Zmin=Zmin, Zmax=Zmax, nx=nx, ny=ny
    )

    # Calculate the current on the old grid
    profiles = eq._profiles
    Jtor = profiles.Jtor(eq.R, eq.Z, eq.psi(), eq.psi_bndry)

    # Interpolate Jtor onto new grid
    Jtor_func = interpolate.RectBivariateSpline(eq.R[:, 0], eq.Z[0, :], Jtor)
    Jtor_new = Jtor_func(result.R, result.Z, grid=False)

    result._applyBoundary(result, Jtor_new, result.plasma_psi)

    # Right hand side of G-S equation
    rhs = -mu0 * result.R * Jtor_new

    # Copy boundary conditions
    rhs[0, :] = result.plasma_psi[0, :]
    rhs[:, 0] = result.plasma_psi[:, 0]
    rhs[-1, :] = result.plasma_psi[-1, :]
    rhs[:, -1] = result.plasma_psi[:, -1]

    # Call elliptic solver
    plasma_psi = result._solver(result.plasma_psi, rhs)

    result._updatePlasmaPsi(plasma_psi)

    # Solve once more, calculating Jtor using new psi
    result.solve(profiles)

    return result

if __name__ == "__main__":

    # Test the different spline interpolation routines

    from numpy import ravel
    import matplotlib.pyplot as plt

    import machine

    tokamak = machine.TestTokamak()

    Rmin = 0.1
    Rmax = 2.0

    eq = Equilibrium(tokamak, Rmin=Rmin, Rmax=Rmax)

    import constraints

    xpoints = [(1.2, -0.8), (1.2, 0.8)]
    constraints.xpointConstrain(eq, xpoints)

    psi = eq.psi()

    tck = interpolate.bisplrep(ravel(eq.R), ravel(eq.Z), ravel(psi))
    spline = interpolate.RectBivariateSpline(eq.R[:, 0], eq.Z[0, :], psi)
    f = interpolate.interp2d(eq.R[:, 0], eq.Z[0, :], psi, kind="cubic")

    plt.plot(eq.R[:, 10], psi[:, 10], "o")

    r = linspace(Rmin, Rmax, 1000)
    z = eq.Z[0, 10]
    plt.plot(r, f(r, z), label="f")

    plt.plot(r, spline(r, z), label="spline")

    plt.plot(r, interpolate.bisplev(r, z, tck), label="bisplev")

    plt.legend()
    plt.show()
