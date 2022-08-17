#!/usr/bin/env python

import freegs
from freegs.machine import Coil
import matplotlib.pyplot as plt

#########################################
# Create the machine, which specifies coil locations
# and equilibrium, specifying the domain to solve over

coils = [
    ("P1L", Coil(1.0, -1.1)),
    ("P1U", Coil(1.0, 1.1)),
    ("P2L", Coil(1.75, -0.6)),
    ("P2U", Coil(1.75, 0.6))
]

# tokamak = freegs.machine.Machine(coils)
tokamak = freegs.machine.TestTokamak()

eq = freegs.Equilibrium(tokamak=tokamak,
                        Rmin=0.1, Rmax=2.0,    # Radial domain
                        Zmin=-1.0, Zmax=1.0,   # Height range
                        nx=65, ny=65,          # Number of grid points
                        boundary=freegs.boundary.freeBoundaryHagenow)  # Boundary condition


#########################################
# Plasma profiles

profiles = freegs.jtor.ConstrainPaxisIp(1e3, # Plasma pressure on axis [Pascals]
                                        2e5, # Plasma current [Amps]
                                        2.0) # Vacuum f=R*Bt

#########################################
# Coil current constraints
#
# Specify locations of the X-points
# to use to constrain the initial value of all coil currents

xpoints = [(1.1, -0.6),   # (R,Z) locations of X-points
           (1.1, 0.8)]

isoflux = [(1.1,-0.6, 1.1,0.6)] # (R1,Z1, R2,Z2) pair of locations

constrain = freegs.control.constrain(xpoints=xpoints, isoflux=isoflux)

#########################################
# Nonlinear solve

freegs.solve(eq,          # The equilibrium to adjust
             profiles,    # The toroidal current profile function
             constrain)   # Constraint function to set coil currents

# eq now contains the solution

# Currents in the coils from the above inverse problem
print("Old coil currents and forces")
print("----------------------------")
tokamak.printCurrents()
eq.printForces()

ax = eq.plot(show=False)
eq.tokamak.plot(axis=ax, show=False)
constrain.plot(axis=ax, show=False)
ax.set_title('Old equilibrium')
plt.show(block=False)

#########################################
# Now you can perturb the coil currents and solve a forward problem
# by calling freegs.solve with constrain=None

for _, coil in eq.tokamak.coils:
    coil.current *= 1.5
    # optionally, you can set
    # coil.control = False
freegs.solve(eq, profiles, constrain=None)

print("\n\nNew coil currents and forces")
print("----------------------------")
tokamak.printCurrents()
eq.printForces()

ax = eq.plot(show=False)
eq.tokamak.plot(axis=ax, show=False)
constrain.plot(axis=ax, show=False)
ax.set_title('New equilibrium')
plt.show(block=False)

input('Press enter to exit')