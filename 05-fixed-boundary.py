#!/usr/bin/env python
#
# Grad-Shafranov solver example
# Fixed boundary (square domain) with no X-points
#

# Plasma equilibrium (Grad-Shafranov) solver
import freegs

# Boundary conditions
import freegs.boundary as boundary

eq = freegs.Equilibrium(Rmin=0.1, Rmax=2.0,
                        Zmin=-1.0, Zmax=1.0,
                        nx=65, ny=65,
                        boundary=boundary.fixedBoundary)

profiles = freegs.jtor.ConstrainPaxisIp(eq,
                                        1e3, # Plasma pressure on axis [Pascals]
                                        1e5, # Plasma current [Amps]
                                        1.0) # fvac = R*Bt

# Nonlinear solver for Grad-Shafranov equation
freegs.solve(eq,           # The equilibrium to adjust
            profiles,	   # The toroidal current profile function
            show = True)

print("Done!")

# Some diagnostics
print("Poloidal beta: {}".format(eq.poloidalBeta()))
print("Pressure on axis: {} Pa".format(eq.pressure(0.0)))

# Plot equilibrium
from freegs.plotting import plotEquilibrium
plotEquilibrium(eq)


