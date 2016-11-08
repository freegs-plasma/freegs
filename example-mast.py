#!/usr/bin/env python

# Options for setting toroidal current
from freegs.jtor import ConstrainPaxisIp

# Plasma equilibrium (Grad-Shafranov) solver
from freegs.equilibrium import Equilibrium

# Coils and current circuits
import freegs.machine as machine

# Control algorithms to constrain the shape,location
import freegs.control as control

# Nonlinear solver
import freegs.picard as picard

#########################################
# Create the machine, which specifies coil locations
# and equilibrium, specifying the domain to solve over

tokamak = machine.MAST()

eq = Equilibrium(tokamak=tokamak,
                 Rmin=0.1, Rmax=2.0,    # Radial domain
                 Zmin=-2.0, Zmax=2.0,   # Height range
                 nx=65, ny=65)        # Number of grid points

#########################################
# Plasma profiles

jtor_func = ConstrainPaxisIp(3e3, # Plasma pressure on axis [Pascals]
                             7e5) # Plasma current [Amps]

#########################################
# Coil current constraints
#
# Specify locations of the X-points
# to use to constrain coil currents

xpoints = [(0.7, -1.1),   # (R,Z) locations of X-points
           (0.7, 1.1)]

isoflux = [(0.7,-1.1, 1.45, 0.0)   # Outboard midplane, lower X-point
           ,(0.7,1.1, 1.45, 0.0)   # Outboard midplane, upper X-point
           ,(0.7,-1.1, 1.5,-1.9)  # Lower X-point, lower outer leg
           ,(0.7,1.1,  1.5, 1.9)  # Upper X-point, upper outer leg
#           ,(0.2,0.0, 1.4,0.0)
#           ,(0.2,-1.4, 0.7,-1.1)
#           ,(0.2,1.4, 0.7, 1.1)
           #,(0.6,-1.1, 0.2, 0.0)
           ]

constrain = lambda eq : control.constrain(eq, xpoints, gamma=1e-12, isoflux=isoflux)

constrain(eq)

#########################################
# Nonlinear solve

picard.solve(eq,           # The equilibrium to adjust
             jtor_func,    # The toroidal current profile function
             constrain, show=True, 
             niter=5, sublevels=5, ncycle=3)    # Constraint function to set coil currents

# Refine using more iterations
#picard.solve(eq,           # The equilibrium to adjust
#             jtor_func,    # The toroidal current profile function
#             constrain, show=False,
#             niter=40, sublevels=5, ncycle=50)

# eq now contains the solution

print("Done!")

print("Plasma current: %e Amps" % (eq.plasmaCurrent()))

tokamak.printCurrents()

##############################################
# Save to geqdsk file

from freegs import geqdsk

with open("mast.geqdsk", "w") as f:
    geqdsk.write(eq, f)

##############################################
# Final plot

from freegs.plotting import plotEquilibrium
plotEquilibrium(eq)

