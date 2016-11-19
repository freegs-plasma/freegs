"""
Grad-Shafranov equation

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

from numpy import zeros

# Elliptic integrals of first and second kind (K and E)
from scipy.special import ellipk, ellipe

from numpy import sqrt, pi, clip

from scipy.sparse import lil_matrix, eye

# Physical constants
mu0 = 4e-7*pi

class GSElliptic:
    """
    Represents the Grad-Shafranov elliptic operator
    
    \Delta^* = R^2 \nabla\cdot\frac{1}{R^2}\nabla
    
    """

    def __init__(self, Rmin):
        """
        Specify minimum radius
        
        """
        self.Rmin = Rmin
        
    def __call__(self, psi, dR, dZ):
        """
        Apply the full operator to 2D field f
        
        Inputs
        ------
        
        psi[R,Z]   A 2D NumPy array containing flux function psi
        dR         A scalar (uniform grid spacing)
        dZ         A scalar (uniform grid spacing)
        
        """
        nx = psi.shape[0]
        ny = psi.shape[1]
        
        b = zeros([nx,ny])

        invdR2 = 1./dR**2
        invdZ2 = 1./dZ**2
        
        for x in range(1,nx-1):
            R = self.Rmin + dR*x  # Major radius of this point
            for y in range(1,ny-1):
                # Loop over points in the domain
                b[x,y] = ( psi[x,y-1]*invdZ2
                           + (invdR2 + 1./(2.*R*dR))*psi[x-1,y]
                           - 2.*(invdR2 + invdZ2) * psi[x,y]
                           + (invdR2 - 1./(2.*R*dR))*psi[x+1,y]
                           + psi[x,y+1]*invdZ2 )
        return b
        
    def diag(self, dR, dZ):
        """
        Return the diagonal elements
        
        """
        return -2./dR**2 - 2./dZ**2

class GSsparse:
    """
    Calculates sparse matrices for the Grad-Shafranov operator
    """
    def __init__(self, Rmin, Rmax, Zmin, Zmax):
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Zmin = Zmin
        self.Zmax = Zmax

    def __call__(self, nx, ny):
        """
        Create a sparse matrix with given resolution
        
        """

        # Calculate grid spacing
        dR = (self.Rmax - self.Rmin)/(nx - 1)
        dZ = (self.Zmax - self.Zmin)/(ny - 1)
        
        # Total number of points
        N = nx * ny

        # Create a linked list sparse matrix
        A = eye(N, format="lil")
        
        invdR2 = 1./dR**2
        invdZ2 = 1./dZ**2
        
        for x in range(1,nx-1):
            R = self.Rmin + dR*x  # Major radius of this point
            for y in range(1,ny-1):
                # Loop over points in the domain
                row = x*ny + y

                # y-1 
                A[row, row-1] = invdZ2

                # x-1
                A[row, row-ny] = (invdR2 + 1./(2.*R*dR))

                # diagonal
                A[row, row] = - 2.*(invdR2 + invdZ2)

                # x+1
                A[row, row+ny] = (invdR2 - 1./(2.*R*dR))

                # y+1
                A[row, row+1] = invdZ2
                
        # Convert to Compressed Sparse Row (CSR) format
        return A.tocsr()
    
def Greens(Rc, Zc, R, Z):
    """
    Calculate poloidal flux at (R,Z) due to a unit current
    at (Rc,Zc) using Greens function
    
    """

    # Calculate k^2
    k2 = 4.*R * Rc / ( (R + Rc)**2 + (Z - Zc)**2 )

    # Clip to between 0 and 1 to avoid nans e.g. when coil is on grid point
    k2 = clip(k2, 1e-10, 1.0 - 1e-10)
    k = sqrt(k2)

    # Note definition of ellipk, ellipe in scipy is K(k^2), E(k^2)
    return (mu0/(2.*pi)) * sqrt(R*Rc) * ( (2. - k2)*ellipk(k2) - 2.*ellipe(k2) ) / k

def GreensBz(Rc, Zc, R, Z, eps=1e-3):
    """
    Calculate radial magnetic field at (R,Z)
    due to unit current at (Rc, Zc)
    
    Bz = (1/R) d psi/dR
    """
    
    return (Greens(Rc, Zc, R+eps, Z) - Greens(Rc, Zc, R-eps, Z))/(2.*eps*R)

def GreensBr(Rc, Zc, R, Z, eps=1e-3):
    """
    Calculate vertical magnetic field at (R,Z)
    due to unit current at (Rc, Zc)
    
    Br = -(1/R) d psi/dZ
    """
    
    return (Greens(Rc, Zc, R, Z-eps) - Greens(Rc, Zc, R, Z+eps))/(2.*eps*R)
