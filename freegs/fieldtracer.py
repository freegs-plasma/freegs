# Field line tracing

from builtins import object

import numpy as np
from scipy.integrate import odeint

class FieldTracer(object):
    """A class for following magnetic field lines
    """
    
    def __init__(self, eq):
        """
        Initialise a FieldTracer with an Equilibrium eq
        """
        self._eq = eq

        # if eq.tokamak.wall:
        #     # Machine has a wall, used to define edges
            
        #     self.insideWalls = self.wallDomain
        # else:
        #     # No wall, so just use the domain
        #     self.insideWalls = self.eqDomain
            
            
    def fieldDirection(self, pos, toroidal_angle):
        """
        Calculate the magnetic field direction at a given pos
        """
        position = pos.reshape((-1, 3))
        R = position[:,0]
        Z = position[:,1]
        # Length is position[:,2]

        # Calculate magnetic field components
        Br = self._eq.Br(R, Z)
        Bz = self._eq.Bz(R, Z)
        Btor = self._eq.Btor(R, Z)
        
        B = np.sqrt(Br**2 + Bz**2 + Btor**2)

        # Rate of change of position with toroidal angle phi
        dRdphi = R * Br / Btor
        dZdphi = R * Bz / Btor
        dldphi = R * B / Btor

        # Detect when the boundary has been reached
        
        
        return np.column_stack((dRdphi, dZdphi, dldphi)).flatten()

    
    def follow(self, Rstart, Zstart, angles, rtol=None):

        Rstart = np.array(Rstart)
        Zstart = np.array(Zstart)
        
        array_shape = Rstart.shape
        assert Zstart.shape == array_shape

        # (R,Z,length) with length=0 initially
        position = np.column_stack((Rstart, Zstart, np.zeros(array_shape))).flatten()
        result = odeint(self.fieldDirection, position, angles, rtol=rtol)

        return result.reshape(angles.shape + array_shape + (3,))
