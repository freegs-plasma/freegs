#!/usr/bin/env python
#
# Example demonstrating functions for creating and finding X-points

import machine
from equilibrium import Equilibrium
import constraints

# Plotting routines
from plotting import plotEquilibrium, plotCoils, plotConstraints

import matplotlib.pyplot as plt

tokamak = machine.TestTokamak()
eq = Equilibrium(tokamak=tokamak, nx=256,ny=256)

##########################################################
# Calculate currents in coils to create X-points
# in specified locations
# 

xpoints = [(1.1, -0.8),   # (R,Z) locations of X-points
           (1.1, 0.8)]

constraints.xpointConstrain(eq, xpoints)

psi = eq.psi()

print("=> Solved coil currents, created X-points")

ax = plotEquilibrium(eq.R, eq.Z, psi)
plotCoils(tokamak.coils, axis=ax)
plotConstraints(xpoints, axis=ax)
plt.show()

##########################################################
# Find critical points (O- and X-points)
# 
# 

import critical
opt, xpt = critical.find_critical(eq.R, eq.Z, psi)

print("=> Found O- and X-points")

ax = plotEquilibrium(eq.R, eq.Z, psi)
for r,z,_ in xpt:
    ax.plot(r,z,'ro')
for r,z,_ in opt:
    ax.plot(r,z,'go')
psi_bndry = xpt[0][2]
sep_contour=ax.contour(eq.R, eq.Z,psi, levels=[psi_bndry], colors='r')
plt.show()

##########################################################
# Create a mask array, 1 in the core and 0 outside
# 
# 

mask = critical.core_mask(eq.R, eq.Z, psi, opt, xpt)

print("=> Created X-point mask")

plt.contourf(eq.R, eq.Z, mask)
plt.show()
