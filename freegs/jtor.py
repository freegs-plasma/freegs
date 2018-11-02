"""
 Classes representing plasma profiles.

 These must have the following methods:

   Jtor(R, Z, psi, psi_bndry=None) 
      -> Return a numpy array of toroidal current density [J/m^2]
   pprime(psinorm)
      -> return p' at given normalised psi
   ffprime(psinorm)
      -> return ff' at given normalised psi
   pressure(psinorm)
      -> return p at given normalised psi
   fpol(psinorm)
      -> return f at given normalised psi
   fvac()
      -> f = R*Bt in vacuum


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

from scipy.integrate import romb, quad # Romberg integration
from . import critical
from .gradshafranov import mu0

from numpy import clip, zeros, reshape, sqrt, pi


class Profile(object):
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
        
        if not hasattr(psinorm, 'shape'):
            # Assume  a single value
            val, _ = quad(self.pprime, psinorm, 1.0)
            # Convert from integral in normalised psi to integral in psi
            return val * (self.psi_axis - self.psi_bndry)
            
        # Assume a NumPy array
        
        if out is None:
            out = zeros(psinorm.shape)
        
        pvals = reshape(psinorm, -1)
        ovals = reshape(out, -1)

        if len(pvals) != len(ovals):
            raise ValueError("Input and output arrays of different lengths")
        
        for i in range(len(pvals)):
            val, _ = quad(self.pprime, pvals[i], 1.0)
            # Convert from integral in normalised psi to integral in psi
            val *= self.psi_axis - self.psi_bndry
            ovals[i] = val
        
        return reshape(ovals, psinorm.shape)
        
    def fpol(self, psinorm, out=None):
        """
        Return f as a function of normalised psi
        
        """
        
        if not hasattr(psinorm, 'shape'):
            # Assume  a single value
            
            val, _ = quad(self.ffprime, psinorm, 1.0)
            # Convert from integral in normalised psi to integral in psi
            val *= self.psi_axis - self.psi_bndry
            
            # ffprime = 0.5*d/dpsi(f^2)
            # Apply boundary condition at psinorm=1 val = fvac**2
            return sqrt(2.*val + self.fvac()**2)
            
        # Assume it's a NumPy array
        
        if out is None:
            out = zeros(psinorm.shape)
            
        pvals = reshape(psinorm, -1)
        ovals = reshape(out, -1)
        
        if len(pvals) != len(ovals):
            raise ValueError("Input and output arrays of different lengths")
        for i in range(len(pvals)):
            val, _ = quad(self.ffprime, pvals[i], 1.0)
            # Convert from integral in normalised psi to integral in psi
            val *= self.psi_axis - self.psi_bndry
            
            # ffprime = 0.5*d/dpsi(f^2)
            # Apply boundary condition at psinorm=1 val = fvac**2
            ovals[i] = sqrt(2.*val + self.fvac()**2)
            
        return reshape(ovals, psinorm.shape)
    
class ConstrainBetapIp(Profile):
    """
    Constrain poloidal Beta and plasma current

    This is the constraint used in
    YoungMu Jeon arXiv:1503.03135
    
    """

    def __init__(self, betap, Ip, fvac, 
                 alpha_m=1.0, alpha_n=2.0,
                 Raxis=1.0):
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

    def Jtor(self, R, Z, psi, psi_bndry=None):
        """ Calculate toroidal plasma current
        
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
            psi_bndry = psi[0,0]
            mask = None
        
        dR = R[1,0] - R[0,0]
        dZ = Z[0,1] - Z[0,0]
        
        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = (psi - psi_axis)  / (psi_bndry - psi_axis)
        
        # Current profile shape
        jtorshape = (1. - psi_norm**self.alpha_m)**self.alpha_n
        
        if mask is not None:
            # If there is a masking function (X-points, limiters)
            jtorshape *= mask
            
        # Now apply constraints to define constants
        
        # Need integral of jtorshape to calculate pressure
        # Note factor to convert from normalised psi integral
        def pshape(psinorm):
            shapeintegral,_ =  quad(lambda x: (1. - x**self.alpha_m)**self.alpha_n, psinorm, 1.0)
            shapeintegral *= (psi_bndry - psi_axis)
            return shapeintegral
        
        # Pressure is
        # 
        # p(psinorm) = - (L*Beta0/Raxis) * pshape(psinorm)
        #
        
        nx,ny = psi_norm.shape
        pfunc = zeros((nx,ny))
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                if (psi_norm[i,j] >= 0.0) and (psi_norm[i,j] < 1.0):
                    pfunc[i,j] = pshape(psi_norm[i,j])
        if mask is not None:
            pfunc *= mask
        
        # Integrate over plasma
        # betap = (8pi/mu0) * int(p)dRdZ / Ip^2
        #       = - (8pi/mu0) * (L*Beta0/Raxis) * intp / Ip^2
        
        intp = romb(romb(pfunc)) * dR*dZ
        
        LBeta0 = -self.betap * (mu0/(8.*pi)) * self.Raxis * self.Ip**2 / intp
        
        # Integrate current components
        IR = romb(romb(jtorshape * R/self.Raxis)) * dR*dZ
        I_R = romb(romb(jtorshape * self.Raxis/R)) * dR*dZ
        
        # Toroidal plasma current Ip is
        #
        # Ip = L * (Beta0 * IR + (1-Beta0)*I_R)
        #    = L*Beta0*(IR - I_R) + L*I_R
        #
        
        L = self.Ip/I_R - LBeta0*(IR/I_R - 1)
        Beta0 = LBeta0 / L

        print("Constraints: L = %e, Beta0 = %e" % (L, Beta0))
        
        # Toroidal current
        Jtor = L * (Beta0*R/self.Raxis + (1-Beta0)*self.Raxis/R)*jtorshape
        
        self.L = L
        self.Beta0 = Beta0
        self.psi_bndry = psi_bndry
        self.psi_axis = psi_axis

        return Jtor

    # Profile functions
    def pprime(self, pn):
        """
        dp/dpsi as a function of normalised psi
        """
        shape = (1. - pn**self.alpha_m)**self.alpha_n
        return self.L * self.Beta0/self.Raxis * shape
        
    def ffprime(self, pn):
        """
        f * df/dpsi as a function of normalised psi
        """
        shape = (1. - pn**self.alpha_m)**self.alpha_n
        return mu0 * self.L * (1-self.Beta0)*self.Raxis * shape
        
        return Jtor, pprime, ffprime
        
    def fvac(self):
        return self._fvac


class ConstrainPaxisIp(Profile):
    """
    Constrain pressure on axis and plasma current
    
    """

    def __init__(self, paxis, Ip, fvac, 
                 alpha_m=1.0, alpha_n=2.0,
                 Raxis=1.0):
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

    def Jtor(self, R, Z, psi, psi_bndry=None):
        """ Calculate toroidal plasma current
        
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
            psi_bndry = psi[0,0]
            mask = None
        
        dR = R[1,0] - R[0,0]
        dZ = Z[0,1] - Z[0,0]
        
        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = (psi - psi_axis)  / (psi_bndry - psi_axis)
        
        # Current profile shape
        jtorshape = (1. - psi_norm**self.alpha_m)**self.alpha_n
        
        if mask is not None:
            # If there is a masking function (X-points, limiters)
            jtorshape *= mask
            
        # Now apply constraints to define constants
        
        # Need integral of jtorshape to calculate paxis
        # Note factor to convert from normalised psi integral
        shapeintegral,_ =  quad(lambda x: (1. - x**self.alpha_m)**self.alpha_n, 0.0, 1.0)
        shapeintegral *= (psi_bndry - psi_axis)
        
        # Pressure on axis is
        # 
        # paxis = - (L*Beta0/Raxis) * shapeintegral
        #

        # Integrate current components
        IR = romb(romb(jtorshape * R/self.Raxis)) * dR*dZ
        I_R = romb(romb(jtorshape * self.Raxis/R)) * dR*dZ
        
        # Toroidal plasma current Ip is
        #
        # Ip = L * (Beta0 * IR + (1-Beta0)*I_R)
        #    = L*Beta0*(IR - I_R) + L*I_R
        #
        
        LBeta0 = -self.paxis*self.Raxis / shapeintegral
        
        L = self.Ip/I_R - LBeta0*(IR/I_R - 1)
        Beta0 = LBeta0 / L

        print("Constraints: L = %e, Beta0 = %e" % (L, Beta0))
        
        # Toroidal current
        Jtor = L * (Beta0*R/self.Raxis + (1-Beta0)*self.Raxis/R)*jtorshape
        
        self.L = L
        self.Beta0 = Beta0
        self.psi_bndry = psi_bndry
        self.psi_axis = psi_axis

        return Jtor

    # Profile functions
    def pprime(self, pn):
        """
        dp/dpsi as a function of normalised psi
        """
        shape = (1. - pn**self.alpha_m)**self.alpha_n
        return self.L * self.Beta0/self.Raxis * shape
        
    def ffprime(self, pn):
        """
        f * df/dpsi as a function of normalised psi
        """
        shape = (1. - pn**self.alpha_m)**self.alpha_n
        return mu0 * self.L * (1-self.Beta0)*self.Raxis * shape
        
        return Jtor, pprime, ffprime
        
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
        self.pprime = pprime_func
        self.ffprime = ffprime_func
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
            psi_bndry = psi[0,0]
            mask = None
        
        dR = R[1,0] - R[0,0]
        dZ = Z[0,1] - Z[0,0]
        
        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = clip((psi - psi_axis)  / (psi_bndry - psi_axis), 0.0, 1.0)
        Jtor = R * self.pprime(psi_norm) + self.ffprime(psi_norm)/(R * mu0)

        if mask is not None:
            # If there is a masking function (X-points, limiters)
            Jtor *= mask
        
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

'''    
class ProfilesSafety:
    """
    Specified profile functions safety factor
    
    """
    def __init__(self, q_func=None):
        """
        q_func = Returns q profile for a normalised value of psi
        
        Optionally, the pres
        """
        self.q_func = q_func
        
   
    
    def q(self, psinorm, out=None):
        """
        Return f=R*Bt at given value(s) of
        normalised psi.
        """
        if self.f_func is not None:
            # If a function exists then use it
            return self.q_func(psinorm)
        
        # If not, use base class to integrate
        return super(ProfilesSafety, self).q(psinorm, out)
'''
