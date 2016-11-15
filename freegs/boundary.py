"""
Functions to impose boundary conditions on psi(plasma)

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

from .gradshafranov import Greens
from numpy import concatenate

from scipy.integrate import romb # Romberg integration

def fixedBoundary(R, Z, Jtor, psi):
    """
    Set psi=0 on all boundaries
    """
    psi[0,:] = 0.0
    psi[:,0] = 0.0
    psi[-1,:] = 0.0
    psi[:,-1] = 0.0

def freeBoundary(R, Z, Jtor, psi):
    """
    Apply a free boundary condition using Green's functions
    
    """
    nx,ny = psi.shape

    dR = R[1,0] - R[0,0]
    dZ = Z[0,1] - Z[0,0]
    
    # List of indices on the boundary
    bndry_indices = concatenate([
        [(x,0) for x in range(nx)],
        [(x,ny-1) for x in range(nx)],
        [(0,y) for y in range(ny)],
        [(nx-1,y) for y in range(ny)]])
    
    for x,y in bndry_indices:
        # Calculate the response of the boundary point
        # to each cell in the plasma domain
        greenfunc = Greens(R, Z, R[x,y], Z[x,y])

        # Prevent infinity/nan by removing (x,y) point
        greenfunc[x,y] = 0.0 
        
        # Integrate over the domain
        psi[x,y] = romb(romb(greenfunc*Jtor))*dR*dZ
