"""
Grad-Shafranov solver example

Fixed boundary (square domain) with no X-points

"""

# Options for setting toroidal current
from jtor import ConstrainPaxisIp

# Plotting routines
from plotting import plotEquilibrium

# Boundary conditions
import boundary

# Plasma equilibrium (Grad-Shafranov) solver
from equilibrium import Equilibrium

from numpy import amin, amax

import matplotlib.pyplot as plt

nx = 65
ny = 65

Rmin = 0.1
Rmax = 2.0

Zmin = -1.0
Zmax = 1.0

jtor_func = ConstrainPaxisIp(1e4, # Plasma pressure on axis [Pascals]
                             1e6) # Plasma current [Amps]


eq = Equilibrium(Rmin=Rmin, Rmax=Rmax,
                 Zmin=Zmin, Zmax=Zmax,
                 nx=nx, ny=ny,
                 boundary=boundary.fixedBoundary)

psi = eq.psi()

#plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

# Start main loop
while True:
    # Copy psi to compare at the end
    psi_last = psi.copy()
        
    # Find psi at the magnetic axis and boundary
        
    psi_min = amin(psi)
    psi_max = amax(psi)
    if abs(psi_min) > abs(psi_max):
        psi_axis = psi_min
        psi_bndry = psi_max
    else:
        psi_axis = psi_max
        psi_bndry = psi_min
        
    print("psi axis = %e, boundary = %e" % (psi_axis, psi_bndry))
    
    # Calculate toroidal current density
    Jtor = jtor_func(eq.R, eq.Z, psi, psi_axis, psi_bndry)
    
    # Solve equilbrium
    eq.solve(Jtor)
    
    # Get the new psi, including coils
    psi = eq.psi()
    
    # Compare against last solution
    psi_last -= psi 
    psi_maxchange = amax(abs(psi_last))
    psi_relchange = psi_maxchange/abs(psi_bndry - psi_axis)
    
    print("Maximum change in psi: %e. Relative: %e" % (psi_maxchange, psi_relchange))
    
    ax.clear()
    plotEquilibrium(eq.R, eq.Z, psi, axis=ax)
    fig.canvas.draw()
    plt.pause(0.0001) 
        
    # Check if the relative change in psi is small enough
    if psi_relchange < 1e-3:
        break

print("Done!")

plt.show()


