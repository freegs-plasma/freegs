#!/usr/bin/env python

import freegs

#########################################
# Create the machine, which specifies coil locations
# and equilibrium, specifying the domain to solve over

tokamak = freegs.machine.MAST()

eq = freegs.Equilibrium(tokamak=tokamak,
                        Rmin=0.1, Rmax=2.0,    # Radial domain
                        Zmin=-2.0, Zmax=2.0,   # Height range
                        nx=65, ny=65)          # Number of grid points

#########################################
# Plasma profiles

profiles = freegs.jtor.ConstrainPaxisIp(3e3, # Plasma pressure on axis [Pascals]
                                        7e5, # Plasma current [Amps]
                                        0.4) # vacuum f = R*Bt

#########################################
# Coil current constraints
#
# Specify locations of the X-points
# to use to constrain coil currents

xpoints = [(0.7, -1.1),   # (R,Z) locations of X-points
           (0.7, 1.1)]

isoflux = [(0.7,-1.1, 1.45, 0.0)   # Outboard midplane, lower X-point
           ,(0.7,1.1, 1.45, 0.0)   # Outboard midplane, upper X-point
#           ,(0.7,-1.1, 1.5,-1.9)  # Lower X-point, lower outer leg
#           ,(0.7,1.1,  1.5, 1.9)  # Upper X-point, upper outer leg
           ]

constrain = freegs.control.constrain(xpoints=xpoints, gamma=1e-12, isoflux=isoflux)

constrain(eq)

#########################################
# Nonlinear solve

freegs.solve(eq,          # The equilibrium to adjust
             profiles,    # The plasma profiles
             constrain,   # Plasma control constraints
             show=True)   # Shows results at each nonlinear iteration

# eq now contains the solution

print("Done!")

print("Plasma current: %e Amps" % (eq.plasmaCurrent()))
print("Pressure on axis: %e Pascals" % (eq.pressure(0.0)))
print("Plasma poloidal beta: %e" % (eq.poloidalBeta()))
print("Plasma volume: %e m^3" % (eq.plasmaVolume()))

eq.tokamak.printCurrents()

##############################################
# Save to geqdsk file

from freegs import geqdsk

with open("mast.geqdsk", "w") as f:
    geqdsk.write(eq, f)

# Call matplotlib show so plot pauses
import matplotlib.pyplot as plt
plt.show()
