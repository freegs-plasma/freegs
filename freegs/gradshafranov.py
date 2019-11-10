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
    
    which is

    d^2/dR^2 + d^2/dZ^2 - (1/R)*d/dR
    
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

class GSsparse4thOrder:
    """
    Calculates sparse matrices for the Grad-Shafranov operator using 4th-order
    finite differences. 
    """

    # Coefficients for first derivatives
    # (index offset, weight)
    
    centred_1st = [(-2,  1./12),
                   (-1, -8./12),
                   ( 1,  8./12),
                   ( 2, -1./12)]
    
    offset_1st = [(-1,  -3./12),
                  ( 0, -10./12),
                  ( 1,  18./12),
                  ( 2,  -6./12),
                  ( 3,   1./12)]
        
    # Coefficients for second derivatives
    # (index offset, weight)
    centred_2nd = [(-2, -1./12),
                   (-1, 16./12),
                   ( 0, -30./12),
                   ( 1, 16./12),
                   ( 2, -1./12)]
    
    offset_2nd = [(-1,  10./12),
                  ( 0, -15./12),
                  ( 1,  -4./12),
                  ( 2,  14./12),
                  ( 3,  -6./12),
                  ( 4,   1./12)]
    
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
        
        # Total number of points, including boundaries
        N = nx * ny

        # Create a linked list sparse matrix
        A = lil_matrix((N,N))
        
        invdR2 = 1./dR**2
        invdZ2 = 1./dZ**2
        
        for x in range(1,nx-1):
            R = self.Rmin + dR*x  # Major radius of this point
            for y in range(1,ny-1):
                row = x*ny + y

                # d^2 / dZ^2
                if y == 1:
                    # One-sided derivatives in Z
                    for offset, weight in self.offset_2nd:
                        A[row, row + offset] += weight * invdZ2
                elif y == ny-2:
                    # One-sided, reversed direction.
                    # Note that for second derivatives the sign of the weights doesn't change
                    for offset, weight in self.offset_2nd:
                        A[row, row - offset] += weight * invdZ2
                else:
                    # Central differencing
                    for offset, weight in self.centred_2nd:
                        A[row, row + offset] += weight * invdZ2

                # d^2 / dR^2 - (1/R) d/dR

                if x == 1:
                    for offset, weight in self.offset_2nd:
                        A[row, row + offset*ny] += weight * invdR2
                        
                    for offset, weight in self.offset_1st:
                        A[row, row + offset*ny] -= weight / (R * dR)
                        
                elif x == nx-2:
                    for offset, weight in self.offset_2nd:
                        A[row, row - offset*ny] += weight * invdR2
                        
                    for offset, weight in self.offset_1st:
                        A[row, row - offset*ny] += weight / (R * dR)
                else:
                    for offset, weight in self.centred_2nd:
                        A[row, row + offset*ny] += weight * invdR2
                        
                    for offset, weight in self.centred_1st:
                        A[row, row + offset*ny] -= weight / (R * dR)

        # Set boundary rows
        for x in range(nx):
            for y in [0, ny-1]:
                row = x*ny + y
                A[row, row] = 1.0
        for x in [0, nx-1]:
            for y in range(ny):
                row = x*ny + y
                A[row, row] = 1.0

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
