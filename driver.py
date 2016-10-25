
import machine
from equilibrium import Equilibrium
import constraints

# Plotting routines
from plotting import plotEquilibrium, plotCoils, plotConstraints

import matplotlib.pyplot as plt

tokamak = machine.TestTokamak()
eq = Equilibrium(tokamak=tokamak, nx=65,ny=65)

xpoints = [(1.1, -0.8),
           (1.1, 0.8)]
constraints.xpointConstrain(eq, tokamak, xpoints)

ax = plotEquilibrium(eq.R, eq.Z, eq.psi())
plotCoils(tokamak.coils, axis=ax)
plotConstraints(xpoints, axis=ax)

plt.show()

import critical

critical.find_critical(eq.R, eq.Z, eq.psi())
