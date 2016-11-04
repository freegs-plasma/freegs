#!/usr/bin/env python

# Options for setting toroidal current
from freegs.jtor import ConstrainPaxisIp

# Plasma equilibrium (Grad-Shafranov) solver
from freegs.equilibrium import Equilibrium

# Coils and current circuits
import freegs.machine as machine

# Control algorithms to constrain the shape,location
import freegs.constraints as constraints

# Nonlinear solver
import freegs.picard as picard

#########################################
# Create the machine, which specifies coil locations
# and equilibrium, specifying the domain to solve over

tokamak = machine.TestTokamak()

eq = Equilibrium(tokamak=tokamak,
                 Rmin=0.1, Rmax=2.0,    # Radial domain
                 Zmin=-1.0, Zmax=1.0,   # Height range
                 nx=65, ny=65)        # Number of grid points

#########################################
# Plasma profiles

jtor_func = ConstrainPaxisIp(1e4, # Plasma pressure on axis [Pascals]
                             1e6) # Plasma current [Amps]

#########################################
# Coil current constraints
#
# Specify locations of the X-points
# to use to constrain coil currents

xpoints = [(1.1, -0.6),   # (R,Z) locations of X-points
           (1.1, 0.8)]

constrain = lambda eq : constraints.xpointConstrain(eq, xpoints)

#########################################
# Nonlinear solve

picard.solve(eq,           # The equilibrium to adjust
             jtor_func,    # The toroidal current profile function
             constrain)    # Constraint function to set coil currents

# eq now contains the solution

print("Done!")

print("Plasma current: %e Amps" % (eq.plasmaCurrent()))

tokamak.printCurrents()

##############################################
# Final plot

from freegs.plotting import plotEquilibrium
plotEquilibrium(eq)
