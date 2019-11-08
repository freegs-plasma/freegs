"""
Routines for solving the nonlinear part of the Grad-Shafranov equation

Copyright 2016-2019 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk

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

from numpy import amin, amax

def solve(eq, profiles, constrain=None, rtol=1e-3, atol=1e-10, blend=0.0,
          show=False, axis=None, pause=0.0001, psi_bndry=None, maxits=50):
    """
    Perform Picard iteration to find solution to the Grad-Shafranov equation
    
    eq       - an Equilibrium object (equilibrium.py)
    profiles - A Profile object for toroidal current (jtor.py)

    rtol     - Relative tolerance (change in psi)/( max(psi) - min(psi) )
    atol     - Absolute tolerance, change in psi
    blend    - Blending of previous and next psi solution
               psi{n+1} <- psi{n+1} * (1-blend) + blend * psi{n}

    show     - If true, plot the plasma equilibrium at each nonlinear step
    axis     - Specify a figure to plot onto. Default (None) creates a new figure
    pause    - Delay between output plots. If negative, waits for window to be closed
    
    maxits   - Maximum number of iterations. Set to None for no limit.
               If this limit is exceeded then a RuntimeError is raised.
    """
    
    if constrain is not None:
        # Set the coil currents to get X-points in desired locations
        constrain(eq)
    
    # Get the total psi = plasma + coils
    psi = eq.psi()
    
    if show:
        import matplotlib.pyplot as plt
        from .plotting import plotEquilibrium
        from . import critical
        
        if pause > 0. and axis is None:
            # No axis specified, so create a new figure
            fig = plt.figure()
            axis = fig.add_subplot(111)

    iteration = 0 # Count number of iterations
    # Start main loop
    while True:
        if show:
            # Plot state of plasma equilibrium
            if pause < 0:
                fig = plt.figure()
                axis = fig.add_subplot(111)
            else:
                axis.clear()
            
            plotEquilibrium(eq,axis=axis,show=False)
            
            if pause < 0:
                # Wait for user to close the window
                plt.show()
            else:
                # Update the canvas and pause
                # Note, a short pause is needed to force drawing update
                axis.figure.canvas.draw()
                plt.pause(pause) 
        
        # Copy psi to compare at the end
        psi_last = psi.copy()
        
        # Solve equilbrium, using the given psi to calculate Jtor
        eq.solve(profiles, psi=psi, psi_bndry=psi_bndry)
        
        # Get the new psi, including coils
        psi = eq.psi()
    
        # Compare against last solution
        psi_change = psi_last - psi 
        psi_maxchange = amax(abs(psi_change))
        psi_relchange = psi_maxchange/(amax(psi) - amin(psi))
        
        # Check if the relative change in psi is small enough
        if (psi_maxchange < atol) or (psi_relchange < rtol):
            break
        
        # Adjust the coil currents
        if constrain is not None:
            constrain(eq)
        
        psi = (1. - blend)*eq.psi() + blend*psi_last

        # Check if the maximum iterations has been exceeded
        iteration += 1
        if maxits and iteration > maxits:
            raise RuntimeError("Picard iteration failed to converge (too many iterations)")
        
