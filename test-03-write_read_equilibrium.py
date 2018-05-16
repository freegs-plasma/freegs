#!/usr/bin/env python3

import freegs

from numpy import allclose
from numpy.linalg import norm

from sys import exit

#########################################
# Create the machine, which specifies coil locations
# and equilibrium, specifying the domain to solve over

tokamak = freegs.machine.MAST_sym()

eq = freegs.Equilibrium(tokamak=tokamak,
                        Rmin=0.1, Rmax=2.0,    # Radial domain
                        Zmin=-1.0, Zmax=1.0,   # Height range
                        nx=65, ny=65,          # Number of grid points
                        boundary=freegs.boundary.freeBoundaryHagenow)  # Boundary condition


#########################################
# Plasma profiles

profiles = freegs.jtor.ConstrainPaxisIp(1e4, # Plasma pressure on axis [Pascals]
                                        1e6, # Plasma current [Amps]
                                        2.0) # Vacuum f=R*Bt

#########################################
# Coil current constraints
#
# Specify locations of the X-points
# to use to constrain coil currents

xpoints = [(1.1, -0.6),   # (R,Z) locations of X-points
           (1.1, 0.8)]

isoflux = [(1.1,-0.6, 1.1,0.6)] # (R1,Z1, R2,Z2) pair of locations

constrain = freegs.control.constrain(xpoints=xpoints, isoflux=isoflux)

#########################################
# Nonlinear solve

freegs.solve(eq,          # The equilibrium to adjust
             profiles,    # The toroidal current profile function
             constrain)   # Constraint function to set coil currents

with freegs.OutputFile("test_readwrite.h5", 'w') as f:
    f.write_equilibrium(eq)

with freegs.OutputFile("test_readwrite.h5", 'r') as f:
    read_eq = f.read_equilibrium()

print("\n---------------------------------------------")
tokamaks_match = tokamak == read_eq.tokamak
print("tokamaks match? ", tokamaks_match)
psis_match = allclose(eq.psi(), read_eq.psi())
print("psi() matches? ", psis_match)
print("l2-norm of difference: ", norm(eq.psi() - read_eq.psi(), ord=2))

if tokamaks_match and psis_match:
    exit(0)
else:
    exit(1)
