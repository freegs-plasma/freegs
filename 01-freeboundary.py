#!/usr/bin/env python

import freegs

#########################################
# Create the machine, which specifies coil locations
# and equilibrium, specifying the domain to solve over

tokamak = freegs.machine.TestTokamak()

eq = freegs.Equilibrium(
    tokamak=tokamak,
    Rmin=0.1,
    Rmax=2.0,  # Radial domain
    Zmin=-1.0,
    Zmax=1.0,  # Height range
    nx=65,
    ny=65,  # Number of grid points
    boundary=freegs.boundary.freeBoundaryHagenow,
)  # Boundary condition


#########################################
# Plasma profiles

profiles = freegs.jtor.ConstrainPaxisIp(
    eq,
    1e3,  # Plasma pressure on axis [Pascals]
    2e5,  # Plasma current [Amps]
    2.0,
)  # Vacuum f=R*Bt

#########################################
# Coil current constraints
#
# Specify locations of the X-points
# to use to constrain coil currents

xpoints = [
    (1.1, -0.6),  # (R,Z) locations of X-points
    (1.1, 0.8),
]

isoflux = [(1.1, -0.6, 1.1, 0.6)]  # (R1,Z1, R2,Z2) pair of locations

constrain = freegs.control.constrain(xpoints=xpoints, isoflux=isoflux)

#########################################
# Nonlinear solve

freegs.solve(
    eq,  # The equilibrium to adjust
    profiles,  # The toroidal current profile function
    constrain,
    show=True,
)  # Constraint function to set coil currents

# eq now contains the solution

print("Done!")

print(f"Plasma current: {eq.plasmaCurrent():e} Amps")
print(f"Plasma pressure on axis: {eq.pressure(0.0):e} Pascals")
print(f"Poloidal beta: {eq.poloidalBeta():e}")

# Currents in the coils
tokamak.printCurrents()

# Forces on the coils
eq.printForces()

print("\nSafety factor:\n\tpsi \t q")
for psi in [0.01, 0.9, 0.95]:
    print(f"\t{psi:.2f}\t{eq.q(psi):.2f}")

##############################################
# Save to G-EQDSK file

from freegs import geqdsk

with open("lsn.geqdsk", "w") as f:
    geqdsk.write(eq, f)

##############################################
# Final plot of equilibrium

axis = eq.plot(show=False)
eq.tokamak.plot(axis=axis, show=False)
constrain.plot(axis=axis, show=True)

# Safety factor

import matplotlib.pyplot as plt

plt.plot(*eq.q())
plt.xlabel(r"Normalised $\psi$")
plt.ylabel("Safety factor")
plt.show()
