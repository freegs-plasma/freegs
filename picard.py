"""
Routines for solving the nonlinear part of the Grad-Shafranov equation
"""

from numpy import amin, amax

def solve(eq, jtor_func, constrain=None, rtol=1e-3, blend=0.0,
          show=False, pause=0.0001, 
          niter=2, sublevels=4, ncycle=2):
    """
    Perform Picard iteration to find solution to the Grad-Shafranov equation
    
    rtol     - Relative tolerance (change in psi)/( max(psi) - min(psi) )
    blend    - Blending of previous and next psi solution
               psi{n+1} <- psi{n+1} * (1-blend) + blend * psi{n}

    The linear solve is controlled by the following parameters:
    
    niter     - Number of Jacobi iterations per level
    sublevels - Number of levels in the multigrid
    ncycle    - Number of V-cycles
    """
    if constrain is not None:
        # Set the coil currents to get X-points in desired locations
        constrain(eq)
    
    psi = eq.psi()
    
    if show:
        import matplotlib.pyplot as plt
        from plotting import plotEquilibrium
        import critical
        
        if pause > 0.:
            fig = plt.figure()
            axis = fig.add_subplot(111)
        
    # Start main loop
    while True:
        if show:
            if pause < 0:
                fig = plt.figure()
                axis = fig.add_subplot(111)
            else:
                axis.clear()
            plotEquilibrium(eq.R, eq.Z, psi, axis=axis)
        
            opt, xpt = critical.find_critical(eq.R, eq.Z, psi)
            psi_bndry = xpt[0][2]
            
            for r,z,_ in xpt:
                axis.plot(r,z,'ro')
            for r,z,_ in opt:
                axis.plot(r,z,'go')
            psi_bndry = xpt[0][2]
            sep_contour=axis.contour(eq.R, eq.Z,psi, levels=[psi_bndry], colors='r')
            
            if pause < 0:
                plt.show()
            else:
                fig.canvas.draw()
                plt.pause(pause) 
        
        # Copy psi to compare at the end
        psi_last = psi.copy()
        
        # Calculate toroidal current density
        Jtor = jtor_func(eq.R, eq.Z, psi)
        
        # Solve equilbrium
        eq.solve(Jtor, niter=niter, sublevels=sublevels, ncycle=ncycle)
        
        # Get the new psi, including coils
        psi = eq.psi()
    
        # Compare against last solution
        psi_last -= psi 
        psi_maxchange = amax(abs(psi_last))
        psi_relchange = psi_maxchange/(amax(psi) - amin(psi))
        
        print("Maximum change in psi: %e. Relative: %e" % (psi_maxchange, psi_relchange))
        
        # Check if the relative change in psi is small enough
        if psi_relchange < rtol:
            break
        
        # Adjust the coil currents
        if constrain is not None:
            constrain(eq)
        
        psi = (1. - blend)*eq.psi() + blend*psi_last
        
        
