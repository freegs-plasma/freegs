# Methods for calculating toroidal current density Jtor
#

from scipy.integrate import romb # Romberg integration

class ConstrainPaxisIp:
    """
    Constrain pressure on axis and plasma current
    """

    def __init__(self, paxis, Ip, alpha_m=1.0, alpha_n=2.0):

        # Check inputs
        if alpha_m < 0:
            raise ValueError("alpha_m must be positive")
        if alpha_n < 0:
            raise ValueError("alpha_n must be positive")

        # Set parameters for later use
        self.paxis = paxis
        self.Ip = Ip
        self.alpha_m = alpha_m
        self.alpha_n = alpha_n

    def __call__(self, R, Z, psi, psi_axis, psi_bndry):
        """ Calculate toroidal plasma current
        
         Jtor = L * (Beta0*R/Rmax + (1-Beta0)*Rmax/R)*jtorshape
        
         where jtorshape is a shape function
         L and Beta0 are parameters which are set by constraints
        """
        
        dR = R[1,0] - R[0,0]
        dZ = Z[0,1] - Z[0,0]
        
        Rmax = R[-1,0]
        
        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = (psi - psi_axis)  / (psi_bndry - psi_axis)
        
        # Current profile shape
        jtorshape = (1. - psi_norm**self.alpha_m)**self.alpha_n
        
        # Now apply constraints to define constants
        # Pressure on axis is
        # 
        # paxis = L*Beta0/Rmax
        #
        # since jtorshape = 1 on axis

        # Integrate current components
        IR = romb(romb(jtorshape * R/Rmax)) * dR*dZ
        I_R = romb(romb(jtorshape * Rmax/R)) * dR*dZ
        
        # Toroidal plasma current Ip is
        #
        # Ip = L * (Beta0 * IR + (1-Beta0)*I_R)
        #    = L*Beta0*(IR - I_R) + L*I_R
        #    = paxis*Rmax*(IR - I_R) + L*I_R
        #

        L = self.Ip/I_R - self.paxis*Rmax*(IR/I_R - 1)
        Beta0 = self.paxis * Rmax / L
        
        print("Constraints: L = %e, Beta0 = %e" % (L, Beta0))

        Jtor = L * (Beta0*R/Rmax + (1-Beta0)*Rmax/R)*jtorshape

        return Jtor
    
