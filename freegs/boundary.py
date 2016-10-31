"""
Functions to impose boundary conditions on psi(plasma)

"""

from gradshafranov import Greens
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
