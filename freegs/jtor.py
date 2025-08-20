"""
Classes representing plasma profiles.

Copyright 2016 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk

This file is part of FreeGS.

FreeGS is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FreeGS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGS.  If not, see <http://www.gnu.org/licenses/>.
"""
from scipy.integrate import romb, quad  # Romberg integration
from scipy.interpolate import interp1d
from . import critical
from .gradshafranov import mu0

from numpy import clip, zeros, reshape, sqrt
import numpy as np
import abc

class Profile(abc.ABC):
    """
    Base class from which profiles classes can inherit

    This provides two methods:
       pressure(psinorm) and fpol(psinorm)

    which assume that the following methods are available:
       pprime(psinorm), ffprime(psinorm), fvac()

    """

    def pressure(self, psinorm, out=None):
        """
        Return p as a function of normalised psi by
        integrating pprime
        """

        if not hasattr(psinorm, "shape"):

            # Assume  a single value
            if hasattr(self,'psi_n_points'):

                # Splined profiles - use prepared spline of pprime integral
                val = self.L * (self.Beta0 / self.Raxis) * self.pprime_int_spline(psinorm)

            else:

                # Parabolic profiles - integrate pprime
                val, _ = quad(self.pprime, psinorm, 1.0)

            # Convert from integral in normalised psi to integral in psi
            val *= -(self.psi_bndry - self.psi_axis)

            return val

        # Assume a NumPy array

        if out is None:
            out = zeros(psinorm.shape)

        pvals = reshape(psinorm, -1)
        ovals = reshape(out, -1)

        if len(pvals) != len(ovals):
            raise ValueError("Input and output arrays of different lengths")

        for i in range(len(pvals)):

            if hasattr(self,'psi_n_points'):

                # Splined profiles - use prepared spline of pprime integral
                val = self.L * self.Beta0 / self.Raxis *self.pprime_int_spline(pvals[i])

            else:

                # Parabolic profiles - integrate pprime
                val, _ = quad(self.pprime, pvals[i], 1.0)

            # Convert from integral in normalised psi to integral in psi
            val *= -(self.psi_bndry - self.psi_axis)
            ovals[i] = val

        return reshape(ovals, psinorm.shape)

    def fpol(self, psinorm, out=None):
        """
        Return f as a function of normalised psi

        """

        if not hasattr(psinorm, "shape"):

            # Assume  a single value
            if hasattr(self,'psi_n_points'):
                
                # Splined profiles - use prepared spline of ffprime integral
                val = self.L * (1-self.Beta0) * self.Raxis *self.ffprime_int_spline(psinorm)

            else:

                # Parabolic profiles - integrate ffprime
                val, _ = quad(self.ffprime, psinorm, 1.0)

            # Convert from integral in normalised psi to integral in psi
            val *= -(self.psi_bndry - self.psi_axis)

            # ffprime = 0.5*d/dpsi(f^2)
            # Apply boundary condition at psinorm=1 val = fvac**2
            return sqrt(2.0 * val + self.fvac() ** 2)

        # Assume it's a NumPy array, or can be converted to one
        psinorm = np.array(psinorm)

        if out is None:
            out = zeros(psinorm.shape)

        pvals = reshape(psinorm, -1)
        ovals = reshape(out, -1)

        if len(pvals) != len(ovals):
            raise ValueError("Input and output arrays of different lengths")
        for i in range(len(pvals)):

            if hasattr(self,'psi_n_points'):
                
                # Splined profiles - use prepared spline of ffprime integral
                val = self.L * (1-self.Beta0) * self.Raxis *self.ffprime_int_spline(pvals[i])

            else:

                # Parabolic profiles - integrate ffprime
                val, _ = quad(self.ffprime, pvals[i], 1.0)

            # Convert from integral in normalised psi to integral in psi
            val *= -(self.psi_bndry - self.psi_axis)

            # ffprime = 0.5*d/dpsi(f^2)
            # Apply boundary condition at psinorm=1 val = fvac**2
            ovals[i] = sqrt(2.0 * val + self.fvac() ** 2)

        return reshape(ovals, psinorm.shape)

    """
    Abstract methods that derived classes must implement.
    """

    @abc.abstractmethod
    def Jtor(
        self, R: np.ndarray, Z: np.ndarray, psi: np.ndarray, psi_bndry=None
    ) -> np.ndarray:
        """Return a numpy array of toroidal current density [J/m^2]"""
        pass

    @abc.abstractmethod
    def pprime(self, psinorm: float) -> float:
        """Return p' at the given normalised psi"""
        pass

    @abc.abstractmethod
    def ffprime(self, psinorm: float) -> float:
        """Return ff' at the given normalised psi"""
        pass

    @abc.abstractmethod
    def fvac(self) -> float:
        """Return f = R*Bt in vacuum"""
        pass

class ConstrainBetapIp(Profile):
    """
    Constrain poloidal Beta and plasma current

    This is the constraint used in
    YoungMu Jeon arXiv:1503.03135

    """

    def __init__(self, eq, betap, Ip, fvac, alpha_m=1.0, alpha_n=2.0, Raxis=1.0):
        """
        betap - Poloidal beta
        Ip    - Plasma current [Amps]
        fvac  - Vacuum f = R*Bt

        Raxis - R used in p' and ff' components
        """

        # Check inputs
        if alpha_m < 0:
            raise ValueError("alpha_m must be positive")
        if alpha_n < 0:
            raise ValueError("alpha_n must be positive")

        # Set parameters for later use
        self.betap = betap
        self.Ip = Ip
        self._fvac = fvac
        self.alpha_m = alpha_m
        self.alpha_n = alpha_n
        self.Raxis = Raxis
        self.eq = eq

    def Jtor(self, R, Z, psi, psi_bndry=None):
        """Calculate toroidal plasma current

        Jtor = L * (Beta0*R/Raxis + (1-Beta0)*Raxis/R)*jtorshape

        where jtorshape is a shape function
        L and Beta0 are parameters which are set by constraints
        """

        # Analyse the equilibrium, finding O- and X-points
        opt, xpt = critical.find_critical(R, Z, psi)
        if not opt:
            raise ValueError("No O-points found!")
        psi_axis = opt[0][2]

        if psi_bndry is not None:
            mask = critical.core_mask(R, Z, psi, opt, xpt, psi_bndry)
        elif xpt:
            psi_bndry = xpt[0][2]
            mask = critical.core_mask(R, Z, psi, opt, xpt)
        else:
            # No X-points
            psi_bndry = psi[0, 0]
            mask = None

        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]

        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)

        # Current profile shape
        jtorshape = (1.0 - np.clip(psi_norm, 0.0, 1.0) ** self.alpha_m) ** self.alpha_n

        if mask is not None:
            # If there is a masking function (X-points, limiters)
            jtorshape *= mask

        # Now apply constraints to define constants

        # Need integral of jtorshape to calculate pressure
        # Note factor to convert from normalised psi integral
        def pshape(psinorm):
            shapeintegral, _ = quad(
                lambda x: (1.0 - x ** self.alpha_m) ** self.alpha_n, psinorm, 1.0
            )
            shapeintegral *= psi_bndry - psi_axis
            return shapeintegral

        # Pressure is
        #
        # p(psinorm) = - (L*Beta0/Raxis) * pshape(psinorm)
        #

        nx, ny = psi_norm.shape
        pfunc = zeros((nx, ny))
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                if (psi_norm[i, j] >= 0.0) and (psi_norm[i, j] < 1.0):
                    pfunc[i, j] = pshape(psi_norm[i, j])
        if mask is not None:
            pfunc *= mask

        # Integrate over plasma
        # betap = (2 mu0) * volume_av(p) / (flux_surf_av(B_poloidal**2))
        #       = - (2 mu0 * L * Beta0 / Raxis) * volume_av(pfunc) / (flux_surf_av(B_poloidal**2))

        p_int = self.eq.calc_volume_averaged(pfunc)
        b_int = self.eq.flux_surface_averaged_Bpol2(psiN=1.0)

        # self.betap = - (2 mu0 * L * Beta0 / Raxis) * (p_int/b_int)
        LBeta0 = (b_int / p_int) * (-self.betap * self.Raxis) / (2 * mu0)

        # Integrate current components
        IR = romb(romb(jtorshape * R / self.Raxis)) * dR * dZ
        I_R = romb(romb(jtorshape * self.Raxis / R)) * dR * dZ

        # Toroidal plasma current Ip is
        #
        # Ip = L * (Beta0 * IR + (1-Beta0)*I_R)
        #    = L*Beta0*(IR - I_R) + L*I_R
        #
        # L = self.Ip / ( (Beta0*IR) + ((1.0-Beta0)*(I_R)) )

        L = self.Ip / I_R - LBeta0 * (IR / I_R - 1)
        Beta0 = LBeta0 / L

        print("Constraints: L = %e, Beta0 = %e" % (L, Beta0))

        # Toroidal current
        Jtor = L * (Beta0 * R / self.Raxis + (1 - Beta0) * self.Raxis / R) * jtorshape

        self.L = L
        self.Beta0 = Beta0

        self.psi_bndry = psi_bndry
        self.psi_axis = psi_axis

        return Jtor

    # Profile functions
    def pprime(self, pn):
        """
        dp/dpsi as a function of normalised psi. 0 outside core
        Calculate pprimeshape inside the core only
        """
        shape = (1.0 - np.clip(pn, 0.0, 1.0) ** self.alpha_m) ** self.alpha_n
        return self.L * self.Beta0 / self.Raxis * shape

    def ffprime(self, pn):
        """
        f * df/dpsi as a function of normalised psi. 0 outside core.
        Calculate ffprimeshape inside the core only.
        """
        shape = (1.0 - np.clip(pn, 0.0, 1.0) ** self.alpha_m) ** self.alpha_n
        return mu0 * self.L * (1 - self.Beta0) * self.Raxis * shape

    def fvac(self):
        return self._fvac


class ConstrainPaxisIp(Profile):
    """
    Constrain pressure on axis and plasma current

    """

    def __init__(self, eq, paxis, Ip, fvac, alpha_m=1.0, alpha_n=2.0, Raxis=1.0):
        """
        paxis - Pressure at magnetic axis [Pa]
        Ip    - Plasma current [Amps]
        fvac  - Vacuum f = R*Bt

        Raxis - R used in p' and ff' components
        """

        # Check inputs
        if alpha_m < 0:
            raise ValueError("alpha_m must be positive")
        if alpha_n < 0:
            raise ValueError("alpha_n must be positive")

        # Set parameters for later use
        self.paxis = paxis
        self.Ip = Ip
        self._fvac = fvac
        self.alpha_m = alpha_m
        self.alpha_n = alpha_n
        self.Raxis = Raxis
        self.eq = eq

    def Jtor(self, R, Z, psi, psi_bndry=None):
        """Calculate toroidal plasma current

        Jtor = L * (Beta0*R/Raxis + (1-Beta0)*Raxis/R)*jtorshape

        where jtorshape is a shape function
        L and Beta0 are parameters which are set by constraints
        """

        # Analyse the equilibrium, finding O- and X-points
        opt, xpt = critical.find_critical(R, Z, psi)
        if not opt:
            raise ValueError("No O-points found!")
        psi_axis = opt[0][2]

        if psi_bndry is not None:
            mask = critical.core_mask(R, Z, psi, opt, xpt, psi_bndry)
        elif xpt:
            psi_bndry = xpt[0][2]
            mask = critical.core_mask(R, Z, psi, opt, xpt)
        else:
            # No X-points
            psi_bndry = psi[0, 0]
            mask = None

        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]

        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)

        # Current profile shape
        jtorshape = (1.0 - np.clip(psi_norm, 0.0, 1.0) ** self.alpha_m) ** self.alpha_n

        if mask is not None:
            # If there is a masking function (X-points, limiters)
            jtorshape *= mask

        # Now apply constraints to define constants

        # Need integral of jtorshape to calculate paxis
        # Note factor to convert from normalised psi integral
        shapeintegral, _ = quad(
            lambda x: (1.0 - x ** self.alpha_m) ** self.alpha_n, 0.0, 1.0
        )
        shapeintegral *= psi_bndry - psi_axis

        # Pressure on axis is
        #
        # paxis = - (L*Beta0/Raxis) * shapeintegral
        #

        LBeta0 = -self.paxis * self.Raxis / shapeintegral

        # Integrate current components
        IR = romb(romb(jtorshape * R / self.Raxis)) * dR * dZ
        I_R = romb(romb(jtorshape * self.Raxis / R)) * dR * dZ

        # Toroidal plasma current Ip is
        #
        # Ip = L * (Beta0 * IR + (1-Beta0)*I_R)
        #    = L*Beta0*(IR - I_R) + L*I_R
        #

        L = self.Ip / I_R - LBeta0 * (IR / I_R - 1)
        Beta0 = LBeta0 / L

        print("Constraints: L = %e, Beta0 = %e" % (L, Beta0))

        # Toroidal current
        Jtor = L * (Beta0 * R / self.Raxis + (1 - Beta0) * self.Raxis / R) * jtorshape

        self.L = L
        self.Beta0 = Beta0
        self.psi_bndry = psi_bndry
        self.psi_axis = psi_axis

        return Jtor

    def pprime(self, pn):
        """
        dp/dpsi as a function of normalised psi. 0 outside core
        Calculate pprimeshape inside the core only
        """
        shape = (1.0 - np.clip(pn, 0.0, 1.0) ** self.alpha_m) ** self.alpha_n
        return self.L * self.Beta0 / self.Raxis * shape
    
    def ffprime(self, pn):
        """
        f * df/dpsi as a function of normalised psi. 0 outside core.
        Calculate ffprimeshape inside the core only.
        """
        shape = (1.0 - np.clip(pn, 0.0, 1.0) ** self.alpha_m) ** self.alpha_n
        return mu0 * self.L * (1 - self.Beta0) * self.Raxis * shape

    def fvac(self):
        return self._fvac

class BetapIpConstrainedSplineProfiles(Profile):
    """
    BetaP and Ip-constrained custom (splined) internal plasma profiles.

    """

    def __init__(self, eq=None, betap=None, Ip=None, Raxis=None, psi_n=None, pprime=None, ffprime=None, fvac=None):
        """
        eq - Equilibrium object
        betap - Poloidal beta
        Ip - Plasma current [Amps]
        Raxis - R used in p' and ff' components
        psi_n - Normalised (0,1) poloidal flux used to defined the profiles
        pprime - Pressure gradient - dp/dpsi
        ffprime - f*dfpol/dpsi
        fvac - Vacuum f = R*Bt

        """

        # Check inputs
        if eq is None:
            raise ValueError("No equilibrium object provided")
        if betap is None:
            raise ValueError("No betap value provided")
        if Ip is None:
            raise ValueError("No plasma current value provided")
        if Raxis is None:
            raise ValueError("No Raxis value provided")
        if psi_n is None:
            raise ValueError("No psi_n data provided")
        if pprime is None:
            raise ValueError("No pprime data provided")
        if ffprime is None:
            raise ValueError("No ffprime data provided")
        if fvac is None:
            raise ValueError("No fvac data provided")

        # Set values for later use
        self.eq = eq
        self.betap = betap
        self.Ip = Ip
        self.Raxis = Raxis
        self.psi_n_points = psi_n
        self.pprime_points = pprime
        self.ffprime_points = ffprime
        self._fvac = fvac

        # Create 1D splines for the internal profiles - these will be like jtorshape
        self.pprime_spline = interp1d(self.psi_n_points,self.pprime_points,kind='linear',fill_value='extrapolate',bounds_error=False)
        self.ffprime_spline = interp1d(self.psi_n_points,self.ffprime_points,kind='linear',fill_value='extrapolate',bounds_error=False)

        # Create 1D splines for the integral of pprime, ffprime
        pn_points = np.linspace(0.0,1.0,100,endpoint=True)

        def pprime_int_func(pn):
            val, _ = quad(self.pprime_spline,pn,1.0)
            return val

        pprime_int_vals = []
        for pn in pn_points:
            pprime_int_vals.append(pprime_int_func(pn))

        pprime_int_vals = np.asarray(pprime_int_vals)

        self.pprime_int_spline = interp1d(pn_points,pprime_int_vals,kind='linear',fill_value='extrapolate',bounds_error=False)

        def ffprime_int_func(pn):
            val, _ = quad(self.ffprime_spline,pn,1.0)
            return val

        ffprime_int_vals = []
        for pn in pn_points:
            ffprime_int_vals.append(ffprime_int_func(pn))

        ffprime_int_vals = np.asarray(ffprime_int_vals)

        self.ffprime_int_spline = interp1d(pn_points,ffprime_int_vals,kind='linear',fill_value='extrapolate',bounds_error=False)

    def Jtor(self, R, Z, psi, psi_bndry=None):
        """Calculate toroidal plasma current

        Jtor = R*pprime + ffprime/(R * mu0)
        """

        # Analyse the equilibrium, finding O- and X-points
        opt, xpt = critical.find_critical(R, Z, psi)
        if not opt:
            raise ValueError("No O-points found!")
        psi_axis = opt[0][2]

        if psi_bndry is not None:
            mask = critical.core_mask(R, Z, psi, opt, xpt, psi_bndry)
        elif xpt:
            psi_bndry = xpt[0][2]
            mask = critical.core_mask(R, Z, psi, opt, xpt)
        else:
            # No X-points
            psi_bndry = psi[0, 0]
            mask = None

        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]

        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)

        pprime_shape = self.pprime_spline(psi_norm)
        ffprime_shape = self.ffprime_spline(psi_norm)
        
        if mask is not None:
            pprime_shape *= mask
            ffprime_shape *= mask

        # Now apply constraints to define constants

        # Need integral of pprime_shape to calculate pressure
        # as p(psinorm) = - (L*Beta0/Raxis) * pshape(psinorm)

        def pshape(psinorm):
            shapeintegral = self.pprime_int_spline(psinorm)
            shapeintegral *= psi_bndry - psi_axis
            return shapeintegral

        nx, ny = psi_norm.shape
        pfunc = zeros((nx, ny))
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                if (psi_norm[i, j] >= 0.0) and (psi_norm[i, j] < 1.0):
                    pfunc[i, j] = pshape(psi_norm[i, j])
        if mask is not None:
            pfunc *= mask

        # Integrate over plasma
        # betap = (2 mu0) * volume_av(p) / (flux_surf_av(B_poloidal**2))
        #       = - (2 mu0 * L * Beta0 / Raxis) * volume_av(pfunc) / (flux_surf_av(B_poloidal**2))
        p_int = self.eq.calc_volume_averaged(pfunc)
        b_int = self.eq.flux_surface_averaged_Bpol2(psiN=1.0)

        # self.betap = - (2 mu0 * L * Beta0 / Raxis) * (p_int/b_int)
        LBeta0 = (b_int / p_int) * (-self.betap * self.Raxis) / (2 * mu0)
        
        # Integrate current components
        IR = romb(romb(pprime_shape * R/self.Raxis)) * dR*dZ # pprime component
        I_R = romb(romb(ffprime_shape * self.Raxis/(R*mu0))) * dR*dZ # ffprime component
        
        # Toroidal plasma current Ip is
        #
        # Ip = L * (Beta0 * IR + (1-Beta0)*I_R)
        #    = L*Beta0*(IR - I_R) + L*I_R
        #
        # L = self.Ip / ( (Beta0*IR) + ((1.0-Beta0)*(I_R)) )

        L = self.Ip/I_R - LBeta0*(IR/I_R - 1)
        Beta0 = LBeta0 / L

        print("Constraints: L = %e, Beta0 = %e" % (L, Beta0))

        # Toroidal current
        Jtor = L *( (pprime_shape * Beta0 * R / self.Raxis) + ((1 - Beta0) * self.Raxis * ffprime_shape/ (R * mu0)) )

        self.L = L
        self.Beta0 = Beta0
        self.psi_bndry = psi_bndry
        self.psi_axis = psi_axis

        return Jtor

    # Profile functions
    def pprime(self, pn):
        """
        dp/dpsi as a function of normalised psi. 0 outside core.
        Calculate pprimeshape inside the core only
        """
        shape = self.pprime_spline(pn)
        return self.L * self.Beta0 / self.Raxis * shape

    def ffprime(self, pn):
        """
        f * df/dpsi as a function of normalised psi. 0 outside core.
        Calculate ffprimeshape inside the core only.
        """
        shape = self.ffprime_spline(pn)
        return self.L * (1 - self.Beta0) * self.Raxis * shape

    def fvac(self):
        return self._fvac

class PaxisIpConstrainedSplineProfiles(Profile):
    """
    Paxis and Ip-constrained custom (splined) internal plasma profiles.

    """

    def __init__(self, eq=None, paxis=None, Ip=None, Raxis=None, psi_n=None, pprime=None, ffprime=None, fvac=None):
        """
        eq - Equilibrium object
        paxis - Pressure at magnetic axis [Pa]
        Ip - Plasma current [Amps]
        Raxis - R used in p' and ff' components
        psi_n - Normalised (0,1) poloidal flux used to defined the profiles
        pprime - Pressure gradient - dp/dpsi
        ffprime - f*dfpol/dpsi
        fvac - Vacuum f = R*Bt

        """

        # Check inputs
        if eq is None:
            raise ValueError("No equilibrium object provided")
        if paxis is None:
            raise ValueError("No paxis value provided")
        if Ip is None:
            raise ValueError("No plasma current value provided")
        if Raxis is None:
            raise ValueError("No Raxis value provided")
        if psi_n is None:
            raise ValueError("No psi_n data provided")
        if pprime is None:
            raise ValueError("No pprime data provided")
        if ffprime is None:
            raise ValueError("No ffprime data provided")
        if fvac is None:
            raise ValueError("No fvac data provided")

        # Set values for later use
        self.eq = eq
        self.paxis = paxis
        self.Ip = Ip
        self.Raxis = Raxis
        self.psi_n_points = psi_n
        self.pprime_points = pprime
        self.ffprime_points = ffprime
        self._fvac = fvac

        # Create 1D splines for the internal profiles - these will be like jtorshape
        self.pprime_spline = interp1d(self.psi_n_points,self.pprime_points,kind='linear',fill_value='extrapolate',bounds_error=False)
        self.ffprime_spline = interp1d(self.psi_n_points,self.ffprime_points,kind='linear',fill_value='extrapolate',bounds_error=False)

        # Create 1D splines for the integral of pprime, ffprime
        pn_points = np.linspace(0.0,1.0,100,endpoint=True)

        def pprime_int_func(pn):
            val, _ = quad(self.pprime_spline,pn,1.0)
            return val

        pprime_int_vals = []
        for pn in pn_points:
            pprime_int_vals.append(pprime_int_func(pn))

        pprime_int_vals = np.asarray(pprime_int_vals)

        self.pprime_int_spline = interp1d(pn_points,pprime_int_vals,kind='linear',fill_value='extrapolate',bounds_error=False)

        def ffprime_int_func(pn):
            val, _ = quad(self.ffprime_spline,pn,1.0)
            return val

        ffprime_int_vals = []
        for pn in pn_points:
            ffprime_int_vals.append(ffprime_int_func(pn))

        ffprime_int_vals = np.asarray(ffprime_int_vals)

        self.ffprime_int_spline = interp1d(pn_points,ffprime_int_vals,kind='linear',fill_value='extrapolate',bounds_error=False)

    def Jtor(self, R, Z, psi, psi_bndry=None):
        """Calculate toroidal plasma current

        Jtor = R*pprime + ffprime/(R * mu0)
        """

        # Analyse the equilibrium, finding O- and X-points
        opt, xpt = critical.find_critical(R, Z, psi)
        if not opt:
            raise ValueError("No O-points found!")
        psi_axis = opt[0][2]

        if psi_bndry is not None:
            mask = critical.core_mask(R, Z, psi, opt, xpt, psi_bndry)
        elif xpt:
            psi_bndry = xpt[0][2]
            mask = critical.core_mask(R, Z, psi, opt, xpt)
        else:
            # No X-points
            psi_bndry = psi[0, 0]
            mask = None

        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]

        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)

        pprime_shape = self.pprime_spline(psi_norm)
        ffprime_shape = self.ffprime_spline(psi_norm)
        
        if mask is not None:
            pprime_shape *= mask
            ffprime_shape *= mask

        # Now apply constraints to define constants

        # Need integral of pprime_shape to calculate pressure
        # as p(psinorm) = - (L*Beta0/Raxis) * pshape(psinorm)

        shapeintegral = self.pprime_int_spline(0.0)
        shapeintegral *= psi_bndry - psi_axis

        # Pressure on axis is
        #
        # paxis = - (L*Beta0/Raxis) * shapeintegral
        #

        LBeta0 = -self.paxis * self.Raxis / shapeintegral
        
        # Integrate current components
        IR = romb(romb(pprime_shape * R/self.Raxis)) * dR*dZ # pprime component
        I_R = romb(romb(ffprime_shape * self.Raxis/(R*mu0))) * dR*dZ # ffprime component
        
        # Toroidal plasma current Ip is
        #
        # Ip = L * (Beta0 * IR + (1-Beta0)*I_R)
        #    = L*Beta0*(IR - I_R) + L*I_R
        #
        # L = self.Ip / ( (Beta0*IR) + ((1.0-Beta0)*(I_R)) )

        L = self.Ip/I_R - LBeta0*(IR/I_R - 1)
        Beta0 = LBeta0 / L

        print("Constraints: L = %e, Beta0 = %e" % (L, Beta0))

        # Toroidal current
        Jtor = L *( (pprime_shape * Beta0 * R / self.Raxis) + ((1 - Beta0) * self.Raxis * ffprime_shape/ (R * mu0)) )

        self.L = L
        self.Beta0 = Beta0

        self.psi_bndry = psi_bndry
        self.psi_axis = psi_axis

        return Jtor

    # Profile functions
    def pprime(self, pn):
        """
        dp/dpsi as a function of normalised psi. 0 outside core.
        Calculate pprimeshape inside the core only
        """
        shape = self.pprime_spline(pn)
        return self.L * self.Beta0 / self.Raxis * shape

    def ffprime(self, pn):
        """
        f * df/dpsi as a function of normalised psi. 0 outside core.
        Calculate ffprimeshape inside the core only.
        """
        shape = self.ffprime_spline(pn)
        return self.L * (1 - self.Beta0) * self.Raxis * shape

    def fvac(self):
        return self._fvac

class PprimeIpConstrainedSplineProfiles(Profile):
    """
    Pprime and Ip-constrained custom (splined) internal plasma profiles.

    """

    def __init__(self, eq=None, Ip=None, Raxis=None, psi_n=None, pprime=None, ffprime=None, fvac=None):
        """
        eq - Equilibrium object
        Ip - Plasma current [Amps]
        Raxis - R used in p' and ff' components
        psi_n - Normalised (0,1) poloidal flux used to defined the profiles
        pprime - Pressure gradient - dp/dpsi
        ffprime - f*dfpol/dpsi
        fvac - Vacuum f = R*Bt

        """

        # Check inputs
        if eq is None:
            raise ValueError("No equilibrium object provided")
        if Ip is None:
            raise ValueError("No plasma current value provided")
        if Raxis is None:
            raise ValueError("No Raxis value provided")
        if psi_n is None:
            raise ValueError("No psi_n data provided")
        if pprime is None:
            raise ValueError("No pprime data provided")
        if ffprime is None:
            raise ValueError("No ffprime data provided")
        if fvac is None:
            raise ValueError("No fvac data provided")

        # Set values for later use
        self.eq = eq
        self.Ip = Ip
        self.Raxis = Raxis
        self.psi_n_points = psi_n
        self.pprime_points = pprime
        self.ffprime_points = ffprime
        self._fvac = fvac

        # Create 1D splines for the internal profiles - these will be like jtorshape
        self.pprime_spline = interp1d(self.psi_n_points,self.pprime_points,kind='linear',fill_value='extrapolate',bounds_error=False)
        self.ffprime_spline = interp1d(self.psi_n_points,self.ffprime_points,kind='linear',fill_value='extrapolate',bounds_error=False)

        # Create 1D splines for the integral of pprime, ffprime
        pn_points = np.linspace(0.0,1.0,100,endpoint=True)

        def pprime_int_func(pn):
            val, _ = quad(self.pprime_spline,pn,1.0)
            return val

        pprime_int_vals = []
        for pn in pn_points:
            pprime_int_vals.append(pprime_int_func(pn))

        pprime_int_vals = np.asarray(pprime_int_vals)

        self.pprime_int_spline = interp1d(pn_points,pprime_int_vals,kind='linear',fill_value='extrapolate',bounds_error=False)

        def ffprime_int_func(pn):
            val, _ = quad(self.ffprime_spline,pn,1.0)
            return val

        ffprime_int_vals = []
        for pn in pn_points:
            ffprime_int_vals.append(ffprime_int_func(pn))

        ffprime_int_vals = np.asarray(ffprime_int_vals)

        self.ffprime_int_spline = interp1d(pn_points,ffprime_int_vals,kind='linear',fill_value='extrapolate',bounds_error=False)

    def Jtor(self, R, Z, psi, psi_bndry=None):
        """Calculate toroidal plasma current

        Jtor = R*pprime + ffprime/(R * mu0)
        """

        # Analyse the equilibrium, finding O- and X-points
        opt, xpt = critical.find_critical(R, Z, psi)
        if not opt:
            raise ValueError("No O-points found!")
        psi_axis = opt[0][2]

        if psi_bndry is not None:
            mask = critical.core_mask(R, Z, psi, opt, xpt, psi_bndry)
        elif xpt:
            psi_bndry = xpt[0][2]
            mask = critical.core_mask(R, Z, psi, opt, xpt)
        else:
            # No X-points
            psi_bndry = psi[0, 0]
            mask = None

        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]

        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)

        pprime_shape = self.pprime_spline(psi_norm)
        ffprime_shape = self.ffprime_spline(psi_norm)
        
        if mask is not None:
            pprime_shape *= mask
            ffprime_shape *= mask

        # Now apply constraints to define constants

        # Constraining pprime = pprime_spline
        # pprime(psinorm) = (LBeta0 / Raxis) * pprime_spline(psinorm)
        # Hence LBeta0 = Raxis
        LBeta0 = self.Raxis
        
        # Integrate current components
        IR = romb(romb(pprime_shape * R/self.Raxis)) * dR*dZ # pprime component
        I_R = romb(romb(ffprime_shape * self.Raxis/(R*mu0))) * dR*dZ # ffprime component
        
        # Toroidal plasma current Ip is
        #
        # Ip = L * (Beta0 * IR + (1-Beta0)*I_R)
        #    = L*Beta0*(IR - I_R) + L*I_R
        #
        # L = self.Ip / ( (Beta0*IR) + ((1.0-Beta0)*(I_R)) )

        L = self.Ip/I_R - LBeta0*(IR/I_R - 1)
        Beta0 = LBeta0 / L

        print("Constraints: L = %e, Beta0 = %e" % (L, Beta0))

        # Toroidal current
        Jtor = L *( (pprime_shape * Beta0 * R / self.Raxis) + ((1 - Beta0) * self.Raxis * ffprime_shape/ (R * mu0)) )

        self.L = L
        self.Beta0 = Beta0

        self.psi_bndry = psi_bndry
        self.psi_axis = psi_axis

        return Jtor

    # Profile functions
    def pprime(self, pn):
        """
        dp/dpsi as a function of normalised psi. 0 outside core.
        Calculate pprimeshape inside the core only
        """
        shape = self.pprime_spline(pn)
        return self.L * self.Beta0 / self.Raxis * shape

    def ffprime(self, pn):
        """
        f * df/dpsi as a function of normalised psi. 0 outside core.
        Calculate ffprimeshape inside the core only.
        """
        shape = self.ffprime_spline(pn)
        return self.L * (1 - self.Beta0) * self.Raxis * shape

    def fvac(self):
        return self._fvac

class BetapFfprimeConstrainedSplineProfiles(Profile):
    """
    BetaP and Ffprime-constrained custom (splined) internal plasma profiles.

    """

    def __init__(self, eq=None, betap=None, Raxis=None, psi_n=None, pprime=None, ffprime=None, fvac=None):
        """
        eq - Equilibrium object
        betap - Poloidal beta
        Raxis - R used in p' and ff' components
        psi_n - Normalised (0,1) poloidal flux used to defined the profiles
        pprime - Pressure gradient - dp/dpsi
        ffprime - f*dfpol/dpsi
        fvac - Vacuum f = R*Bt

        """

        # Check inputs
        if eq is None:
            raise ValueError("No equilibrium object provided")
        if betap is None:
            raise ValueError("No betap value provided")
        if Raxis is None:
            raise ValueError("No Raxis value provided")
        if psi_n is None:
            raise ValueError("No psi_n data provided")
        if pprime is None:
            raise ValueError("No pprime data provided")
        if ffprime is None:
            raise ValueError("No ffprime data provided")
        if fvac is None:
            raise ValueError("No fvac data provided")

        # Set values for later use
        self.eq = eq
        self.betap = betap
        self.Raxis = Raxis
        self.psi_n_points = psi_n
        self.pprime_points = pprime
        self.ffprime_points = ffprime
        self._fvac = fvac

        # Create 1D splines for the internal profiles - these will be like jtorshape
        self.pprime_spline = interp1d(self.psi_n_points,self.pprime_points,kind='linear',fill_value='extrapolate',bounds_error=False)
        self.ffprime_spline = interp1d(self.psi_n_points,self.ffprime_points,kind='linear',fill_value='extrapolate',bounds_error=False)

        # Create 1D splines for the integral of pprime, ffprime
        pn_points = np.linspace(0.0,1.0,100,endpoint=True)

        def pprime_int_func(pn):
            val, _ = quad(self.pprime_spline,pn,1.0)
            return val

        pprime_int_vals = []
        for pn in pn_points:
            pprime_int_vals.append(pprime_int_func(pn))

        pprime_int_vals = np.asarray(pprime_int_vals)

        self.pprime_int_spline = interp1d(pn_points,pprime_int_vals,kind='linear',fill_value='extrapolate',bounds_error=False)

        def ffprime_int_func(pn):
            val, _ = quad(self.ffprime_spline,pn,1.0)
            return val

        ffprime_int_vals = []
        for pn in pn_points:
            ffprime_int_vals.append(ffprime_int_func(pn))

        ffprime_int_vals = np.asarray(ffprime_int_vals)

        self.ffprime_int_spline = interp1d(pn_points,ffprime_int_vals,kind='linear',fill_value='extrapolate',bounds_error=False)

    def Jtor(self, R, Z, psi, psi_bndry=None):
        """Calculate toroidal plasma current

        Jtor = R*pprime + ffprime/(R * mu0)
        """

        # Analyse the equilibrium, finding O- and X-points
        opt, xpt = critical.find_critical(R, Z, psi)
        if not opt:
            raise ValueError("No O-points found!")
        psi_axis = opt[0][2]

        if psi_bndry is not None:
            mask = critical.core_mask(R, Z, psi, opt, xpt, psi_bndry)
        elif xpt:
            psi_bndry = xpt[0][2]
            mask = critical.core_mask(R, Z, psi, opt, xpt)
        else:
            # No X-points
            psi_bndry = psi[0, 0]
            mask = None

        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]

        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)

        pprime_shape = self.pprime_spline(psi_norm)
        ffprime_shape = self.ffprime_spline(psi_norm)
        
        if mask is not None:
            pprime_shape *= mask
            ffprime_shape *= mask

        # Now apply constraints to define constants

        # Need integral of pprime_shape to calculate pressure
        # as p(psinorm) = - (L*Beta0/Raxis) * pshape(psinorm)

        def pshape(psinorm):
            shapeintegral = self.pprime_int_spline(psinorm)
            shapeintegral *= psi_bndry - psi_axis
            return shapeintegral

        nx, ny = psi_norm.shape
        pfunc = zeros((nx, ny))
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                if (psi_norm[i, j] >= 0.0) and (psi_norm[i, j] < 1.0):
                    pfunc[i, j] = pshape(psi_norm[i, j])
        if mask is not None:
            pfunc *= mask

        # Integrate over plasma
        # betap = (2 mu0) * volume_av(p) / (flux_surf_av(B_poloidal**2))
        #       = - (2 mu0 * L * Beta0 / Raxis) * volume_av(pfunc) / (flux_surf_av(B_poloidal**2))

        p_int = self.eq.calc_volume_averaged(pfunc)
        b_int = self.eq.flux_surface_averaged_Bpol2(psiN=1.0)

        # self.betap = - (2 mu0 * L * Beta0 / Raxis) * (p_int/b_int)
        LBeta0 = (b_int / p_int) * (-self.betap * self.Raxis) / (2 * mu0)

        # Constrain Ffprime = ffprime_spline
        # Ffprime = L*(1-Beta0)*Raxis*ffprime_spline
        # L*(1-Beta0)*Raxis = 1
        # L = LBeta0 + (1.0/Raxis)

        L = LBeta0 + (1.0/self.Raxis)
        Beta0 = LBeta0 / L

        print("Constraints: L = %e, Beta0 = %e" % (L, Beta0))

        # Toroidal current
        Jtor = L *( (pprime_shape * Beta0 * R / self.Raxis) + ((1 - Beta0) * self.Raxis * ffprime_shape/ (R * mu0)) )

        self.L = L
        self.Beta0 = Beta0

        self.psi_bndry = psi_bndry
        self.psi_axis = psi_axis

        return Jtor

    # Profile functions
    def pprime(self, pn):
        """
        dp/dpsi as a function of normalised psi. 0 outside core.
        Calculate pprimeshape inside the core only
        """
        shape = self.pprime_spline(pn)
        return self.L * self.Beta0 / self.Raxis * shape

    def ffprime(self, pn):
        """
        f * df/dpsi as a function of normalised psi. 0 outside core.
        Calculate ffprimeshape inside the core only.
        """
        shape = self.ffprime_spline(pn)
        return self.L * (1 - self.Beta0) * self.Raxis * shape

    def fvac(self):
        return self._fvac

class PaxisFfprimeConstrainedSplineProfiles(Profile):
    """
    Paxis and Ffprime-constrained custom (splined) internal plasma profiles.

    """

    def __init__(self, eq=None, paxis=None, Raxis=None, psi_n=None, pprime=None, ffprime=None, fvac=None):
        """
        eq - Equilibrium object
        betap - Poloidal beta
        Raxis - R used in p' and ff' components
        psi_n - Normalised (0,1) poloidal flux used to defined the profiles
        pprime - Pressure gradient - dp/dpsi
        ffprime - f*dfpol/dpsi
        fvac - Vacuum f = R*Bt

        """

        # Check inputs
        if eq is None:
            raise ValueError("No equilibrium object provided")
        if paxis is None:
            raise ValueError("No paxis value provided")
        if Raxis is None:
            raise ValueError("No Raxis value provided")
        if psi_n is None:
            raise ValueError("No psi_n data provided")
        if pprime is None:
            raise ValueError("No pprime data provided")
        if ffprime is None:
            raise ValueError("No ffprime data provided")
        if fvac is None:
            raise ValueError("No fvac data provided")

        # Set values for later use
        self.eq = eq
        self.paxis = paxis
        self.Raxis = Raxis
        self.psi_n_points = psi_n
        self.pprime_points = pprime
        self.ffprime_points = ffprime
        self._fvac = fvac

        # Create 1D splines for the internal profiles - these will be like jtorshape
        self.pprime_spline = interp1d(self.psi_n_points,self.pprime_points,kind='linear',fill_value='extrapolate',bounds_error=False)
        self.ffprime_spline = interp1d(self.psi_n_points,self.ffprime_points,kind='linear',fill_value='extrapolate',bounds_error=False)

        # Create 1D splines for the integral of pprime, ffprime
        pn_points = np.linspace(0.0,1.0,100,endpoint=True)

        def pprime_int_func(pn):
            val, _ = quad(self.pprime_spline,pn,1.0)
            return val

        pprime_int_vals = []
        for pn in pn_points:
            pprime_int_vals.append(pprime_int_func(pn))

        pprime_int_vals = np.asarray(pprime_int_vals)

        self.pprime_int_spline = interp1d(pn_points,pprime_int_vals,kind='linear',fill_value='extrapolate',bounds_error=False)

        def ffprime_int_func(pn):
            val, _ = quad(self.ffprime_spline,pn,1.0)
            return val

        ffprime_int_vals = []
        for pn in pn_points:
            ffprime_int_vals.append(ffprime_int_func(pn))

        ffprime_int_vals = np.asarray(ffprime_int_vals)

        self.ffprime_int_spline = interp1d(pn_points,ffprime_int_vals,kind='linear',fill_value='extrapolate',bounds_error=False)

    def Jtor(self, R, Z, psi, psi_bndry=None):
        """Calculate toroidal plasma current

        Jtor = R*pprime + ffprime/(R * mu0)
        """

        # Analyse the equilibrium, finding O- and X-points
        opt, xpt = critical.find_critical(R, Z, psi)
        if not opt:
            raise ValueError("No O-points found!")
        psi_axis = opt[0][2]

        if psi_bndry is not None:
            mask = critical.core_mask(R, Z, psi, opt, xpt, psi_bndry)
        elif xpt:
            psi_bndry = xpt[0][2]
            mask = critical.core_mask(R, Z, psi, opt, xpt)
        else:
            # No X-points
            psi_bndry = psi[0, 0]
            mask = None

        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]

        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)

        pprime_shape = self.pprime_spline(psi_norm)
        ffprime_shape = self.ffprime_spline(psi_norm)
        
        if mask is not None:
            pprime_shape *= mask
            ffprime_shape *= mask

        # Now apply constraints to define constants

        # Need integral of pprime_shape to calculate pressure
        # as p(psinorm) = - (L*Beta0/Raxis) * pshape(psinorm)

        shapeintegral = self.pprime_int_spline(0.0)
        shapeintegral *= psi_bndry - psi_axis

        # Pressure on axis is
        #
        # paxis = - (L*Beta0/Raxis) * shapeintegral
        #

        LBeta0 = -self.paxis * self.Raxis / shapeintegral
        
        # Constrain Ffprime = ffprime_spline
        # Ffprime = L*(1-Beta0)*Raxis*ffprime_spline
        # L*(1-Beta0)*Raxis = 1
        # L = LBeta0 + (1.0/Raxis)

        L = LBeta0 + (1.0/self.Raxis)
        Beta0 = LBeta0 / L

        print("Constraints: L = %e, Beta0 = %e" % (L, Beta0))

        # Toroidal current
        Jtor = L *( (pprime_shape * Beta0 * R / self.Raxis) + ((1 - Beta0) * self.Raxis * ffprime_shape/ (R * mu0)) )

        self.L = L
        self.Beta0 = Beta0

        self.psi_bndry = psi_bndry
        self.psi_axis = psi_axis

        return Jtor

    # Profile functions
    def pprime(self, pn):
        """
        dp/dpsi as a function of normalised psi. 0 outside core.
        Calculate pprimeshape inside the core only
        """
        shape = self.pprime_spline(pn)
        return self.L * self.Beta0 / self.Raxis * shape

    def ffprime(self, pn):
        """
        f * df/dpsi as a function of normalised psi. 0 outside core.
        Calculate ffprimeshape inside the core only.
        """
        shape = self.ffprime_spline(pn)
        return self.L * (1 - self.Beta0) * self.Raxis * shape

    def fvac(self):
        return self._fvac

class PprimeFfprimeConstrainedSplineProfiles(Profile):
    """
    Pprime and Ffprime-constrained custom (splined) internal plasma profiles.

    """

    def __init__(self, eq=None, Raxis=None, psi_n=None, pprime=None, ffprime=None, fvac=None):
        """
        eq - Equilibrium object
        Raxis - R used in p' and ff' components
        psi_n - Normalised (0,1) poloidal flux used to defined the profiles
        pprime - Pressure gradient - dp/dpsi
        ffprime - f*dfpol/dpsi
        fvac - Vacuum f = R*Bt

        """

        # Check inputs
        if eq is None:
            raise ValueError("No equilibrium object provided")
        if Raxis is None:
            raise ValueError("No Raxis value provided")
        if psi_n is None:
            raise ValueError("No psi_n data provided")
        if pprime is None:
            raise ValueError("No pprime data provided")
        if ffprime is None:
            raise ValueError("No ffprime data provided")
        if fvac is None:
            raise ValueError("No fvac data provided")

        # Set values for later use
        self.eq = eq
        self.Raxis = Raxis
        self.psi_n_points = psi_n
        self.pprime_points = pprime
        self.ffprime_points = ffprime
        self._fvac = fvac

        # Create 1D splines for the internal profiles - these will be like jtorshape
        self.pprime_spline = interp1d(self.psi_n_points,self.pprime_points,kind='linear',fill_value='extrapolate',bounds_error=False)
        self.ffprime_spline = interp1d(self.psi_n_points,self.ffprime_points,kind='linear',fill_value='extrapolate',bounds_error=False)

        # Create 1D splines for the integral of pprime, ffprime
        pn_points = np.linspace(0.0,1.0,100,endpoint=True)

        def pprime_int_func(pn):
            val, _ = quad(self.pprime_spline,pn,1.0)
            return val

        pprime_int_vals = []
        for pn in pn_points:
            pprime_int_vals.append(pprime_int_func(pn))

        pprime_int_vals = np.asarray(pprime_int_vals)

        self.pprime_int_spline = interp1d(pn_points,pprime_int_vals,kind='linear',fill_value='extrapolate',bounds_error=False)

        def ffprime_int_func(pn):
            val, _ = quad(self.ffprime_spline,pn,1.0)
            return val

        ffprime_int_vals = []
        for pn in pn_points:
            ffprime_int_vals.append(ffprime_int_func(pn))

        ffprime_int_vals = np.asarray(ffprime_int_vals)

        self.ffprime_int_spline = interp1d(pn_points,ffprime_int_vals,kind='linear',fill_value='extrapolate',bounds_error=False)

    def Jtor(self, R, Z, psi, psi_bndry=None):
        """Calculate toroidal plasma current

        Jtor = R*pprime + ffprime/(R * mu0)
        """

        # Analyse the equilibrium, finding O- and X-points
        opt, xpt = critical.find_critical(R, Z, psi)
        if not opt:
            raise ValueError("No O-points found!")
        psi_axis = opt[0][2]

        if psi_bndry is not None:
            mask = critical.core_mask(R, Z, psi, opt, xpt, psi_bndry)
        elif xpt:
            psi_bndry = xpt[0][2]
            mask = critical.core_mask(R, Z, psi, opt, xpt)
        else:
            # No X-points
            psi_bndry = psi[0, 0]
            mask = None

        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]

        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)

        pprime_shape = self.pprime_spline(psi_norm)
        ffprime_shape = self.ffprime_spline(psi_norm)
        
        if mask is not None:
            pprime_shape *= mask
            ffprime_shape *= mask

        # Now apply constraints to define constants

        LBeta0 = self.Raxis
        L = self.Raxis + (1.0/self.Raxis)
        Beta0 = LBeta0/L

        print("Constraints: L = %e, Beta0 = %e" % (L, Beta0))

        # Toroidal current
        Jtor = L *( (pprime_shape * Beta0 * R / self.Raxis) + ((1 - Beta0) * self.Raxis * ffprime_shape/ (R * mu0)) )

        self.L = L
        self.Beta0 = Beta0

        self.psi_bndry = psi_bndry
        self.psi_axis = psi_axis

        return Jtor

    # Profile functions
    def pprime(self, pn):
        """
        dp/dpsi as a function of normalised psi. 0 outside core.
        Calculate pprimeshape inside the core only
        """
        shape = self.pprime_spline(pn)
        return self.L * self.Beta0 / self.Raxis * shape

    def ffprime(self, pn):
        """
        f * df/dpsi as a function of normalised psi. 0 outside core.
        Calculate ffprimeshape inside the core only.
        """
        shape = self.ffprime_spline(pn)
        return self.L * (1 - self.Beta0) * self.Raxis * shape

    def fvac(self):
        return self._fvac
    
class ProfilesPprimeFfprime:
    """
    Specified profile functions p'(psi), ff'(psi)

    Jtor = R*p' + ff'/(R*mu0)

    """

    def __init__(self, pprime_func, ffprime_func, fvac, p_func=None, f_func=None):
        """
        pprime_func(psi_norm) - A function which returns dp/dpsi at given normalised flux
        ffprime_func(psi_norm) - A function which returns f*df/dpsi at given normalised flux (f = R*Bt)

        fvac - Vacuum f = R*Bt

        Optionally, the pres
        """
        self._pprime = pprime_func
        self._ffprime = ffprime_func
        self.p_func = p_func
        self.f_func = f_func
        self._fvac = fvac

    def Jtor(self, R, Z, psi, psi_bndry=None):
        """
        Calculate toroidal plasma current

        Jtor = R*p' + ff'/(R*mu0)
        """

        # Analyse the equilibrium, finding O- and X-points
        opt, xpt = critical.find_critical(R, Z, psi)
        if not opt:
            raise ValueError("No O-points found!")
        psi_axis = opt[0][2]

        if psi_bndry is not None:
            mask = critical.core_mask(R, Z, psi, opt, xpt, psi_bndry)
        elif xpt:
            psi_bndry = xpt[0][2]
            mask = critical.core_mask(R, Z, psi, opt, xpt)
        else:
            # No X-points
            psi_bndry = psi[0, 0]
            mask = None

        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = clip((psi - psi_axis) / (psi_bndry - psi_axis), 0.0, 1.0)
        Jtor = R * self.pprime(psi_norm) + self.ffprime(psi_norm) / (R * mu0)

        if mask is not None:
            # If there is a masking function (X-points, limiters)
            Jtor *= mask

        self.psi_bndry = psi_bndry
        self.psi_axis = psi_axis

        return Jtor

    def pressure(self, psinorm, out=None):
        """
        Return pressure [Pa] at given value(s) of
        normalised psi.
        """
        if self.p_func is not None:
            # If a function exists then use it
            return self.p_func(psinorm)

        # If not, use base class to integrate
        return super(ProfilesPprimeFfprime, self).pressure(psinorm, out)

    def fpol(self, psinorm, out=None):
        """
        Return f=R*Bt at given value(s) of
        normalised psi.
        """
        if self.f_func is not None:
            # If a function exists then use it
            return self.f_func(psinorm)

        # If not, use base class to integrate
        return super(ProfilesPprimeFfprime, self).fpol(psinorm, out)

    def fvac(self):
        return self._fvac

    def pprime(self, psinorm):
        return self._pprime(psinorm)

    def ffprime(self, psinorm):
        return self._ffprime(psinorm)