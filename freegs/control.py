"""
Plasma control system

Use constraints to adjust coil currents
"""

from numpy import dot, transpose, eye, array
from numpy.linalg import inv

def constrain(eq, xpoints=[], gamma=1e-12, isoflux=[], psivals=[]):
    
    """
    Adjust coil currents using constraints
    
    xpoints - A list of X-point (R,Z) locations
    
    isoflux - A list of tuples (R1,Z1, R2,Z2) 
    
    psivals - A list of (R,Z,psi) values

    At least one constraint must be included
    
    gamma - A scalar, minimises the magnitude of the coil currents
    
    """
    
    tokamak = eq.getMachine()
    
    constraint_matrix = []
    constraint_rhs = []
    for xpt in xpoints:
        # Each x-point introduces two constraints
        # 1) Br = 0

        Br = eq.Br(xpt[0], xpt[1])
        
        # Add currents to cancel out this field
        constraint_rhs.append(-Br)
        constraint_matrix.append(
            tokamak.controlBr(xpt[0], xpt[1]))
        
        # 2) Bz = 0
            
        Bz = eq.Bz(xpt[0], xpt[1])
        
        # Add currents to cancel out this field
        constraint_rhs.append(-Bz)
        constraint_matrix.append(
            tokamak.controlBz(xpt[0], xpt[1]))

    # Constrain points to have the same flux
    for r1,z1, r2,z2 in isoflux:
        # Get Psi at (r1,z1) and (r2,z2)
        p1 = eq.psiRZ(r1,z1)
        p2 = eq.psiRZ(r2,z2)
        constraint_rhs.append(p2 - p1)
        
        # Coil responses
        c1 = tokamak.controlPsi(r1,z1)
        c2 = tokamak.controlPsi(r2,z2)
        # Control for the difference between p1 and p2
        c = [ c1val - c2val for c1val, c2val in zip(c1,c2)]
        constraint_matrix.append( c )
    
    # Constrain the value of psi
    for r,z,psi in psivals:
        p1 = eq.psiRZ(r,z)
        constraint_rhs.append(psi - p1)

        # Coil responses
        c = tokamak.controlPsi(r,z)
        constraint_matrix.append( c )

    if not constraint_rhs:
        raise ValueError("No constraints given")

    # Constraint matrix
    A = array(constraint_matrix)
    b = array(constraint_rhs)
    
    # Solve by Tikhonov regularisation
    # minimise || Ax - b ||^2 + ||gamma x ||^2
    #
    # x = (A^T A + gamma^2 I)^{-1}A^T b
    
    # Number of controls (length of x)
    ncontrols = A.shape[1]
        
    # Calculate the change in coil current
    current_change =  dot( inv(dot(transpose(A), A) + gamma**2 * eye(ncontrols)), 
                           dot(transpose(A),b))
    print("Current changes: " + str(current_change))
    tokamak.controlAdjust(current_change)
