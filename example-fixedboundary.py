#!/usr/bin/env python
#
# Grad-Shafranov solver example
# Fixed boundary (square domain) with no X-points
#

# Boundary conditions
import freegs.boundary as boundary

# Plasma equilibrium (Grad-Shafranov) solver
import freegs

jtor_func = freegs.jtor.ConstrainPaxisIp(1e4, # Plasma pressure on axis [Pascals]
                                         1e6) # Plasma current [Amps]

eq = freegs.Equilibrium(Rmin=0.1, Rmax=2.0,
                        Zmin=-1.0, Zmax=1.0,
                        nx=65, ny=65,
                        boundary=boundary.fixedBoundary)

# Nonlinear solver for Grad-Shafranov equation
freegs.solve(eq,           # The equilibrium to adjust
             jtor_func)    # The toroidal current profile function

print("Done!")


# Plot equilibrium
from freegs.plotting import plotEquilibrium

plotEquilibrium(eq)


