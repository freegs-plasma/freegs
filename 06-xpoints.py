#!/usr/bin/env python
#
# Example demonstrating functions for creating and finding X-points

import matplotlib.pyplot as plt

import freegs

# Plotting routines
from freegs.plotting import plotCoils, plotConstraints, plotEquilibrium

tokamak = freegs.machine.TestTokamak()
eq = freegs.Equilibrium(tokamak=tokamak, nx=256,ny=256)

##########################################################
# Calculate currents in coils to create X-points
# in specified locations
# 

xpoints = [(1.1, -0.8),   # (R,Z) locations of X-points
           (1.1, 0.8)]

control = freegs.control.constrain(xpoints=xpoints)
control(eq)  # Apply control to Equilibrium eq

psi = eq.psi()

print("=> Solved coil currents, created X-points")

ax = plotEquilibrium(eq, show=False)
plotCoils(tokamak.coils, axis=ax)
plotConstraints(control, axis=ax)
plt.show()

##########################################################
# Find critical points (O- and X-points)
# 
# 

from freegs import critical

opt, xpt = critical.find_critical(eq.R, eq.Z, psi)

print("=> Found O- and X-points")

ax = plotEquilibrium(eq, show=False, oxpoints=False)
for r,z,_ in xpt:
    ax.plot(r,z,'ro')
for r,z,_ in opt:
    ax.plot(r,z,'go')
psi_bndry = xpt[0][2]
sep_contour = ax.contour(eq.R, eq.Z,psi, levels=[psi_bndry], colors='r')
psi_bndry = eq.psi_bndry
sep_contour = ax.contour(eq.R, eq.Z,psi, levels=[psi_bndry], colors='k')
plt.show()

##########################################################
# Create a mask array, 1 in the core and 0 outside
# 
# 

mask = critical.core_mask(eq.R, eq.Z, psi, opt, xpt)

print("=> Created X-point mask")

plt.contourf(eq.R, eq.Z, mask)
plt.show()
