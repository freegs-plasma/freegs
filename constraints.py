"""
Use constraints to adjust coil currents
"""

from numpy import dot, transpose, eye, array
from numpy.linalg import inv

def xpointConstrain(eq, xpoints, gamma=0.0):
    
    """
    Adjust coil currents using x-point constraints
    
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
