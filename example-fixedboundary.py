#!/usr/bin/env python
#
# Grad-Shafranov solver example
# Fixed boundary (square domain) with no X-points
#

# Options for setting toroidal current
from jtor import ConstrainPaxisIp

# Boundary conditions
import boundary

# Plasma equilibrium (Grad-Shafranov) solver
from equilibrium import Equilibrium

# Nonlinear solver for Grad-Shafranov equation
import picard

jtor_func = ConstrainPaxisIp(1e4, # Plasma pressure on axis [Pascals]
                             1e6) # Plasma current [Amps]

eq = Equilibrium(Rmin=0.1, Rmax=2.0,
                 Zmin=-1.0, Zmax=1.0,
                 nx=65, ny=65,
                 boundary=boundary.fixedBoundary)

picard.solve(eq,           # The equilibrium to adjust
             jtor_func)    # The toroidal current profile function

print("Done!")


# Plotting routines
import matplotlib.pyplot as plt
from plotting import plotEquilibrium

plotEquilibrium(eq.R, eq.Z, eq.psi())
plt.show()


