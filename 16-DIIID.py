#!/usr/bin/env python

import matplotlib.pyplot as plt

import freegs

#########################################
# Create the machine, which specifies coil locations
# and equilibrium, specifying the domain to solve over

tokamak = freegs.machine.DIIID()

eq = freegs.Equilibrium(
    tokamak=tokamak,
    Rmin=0.1,
    Rmax=2.8,  # Radial domain
    Zmin=-1.8,
    Zmax=1.8,  # Height range
    nx=129,
    ny=129,
)  # Number of grid points

#########################################
# Plasma profiles

profiles = freegs.jtor.ConstrainPaxisIp(
    eq,
    159811,  # Plasma pressure on axis [Pascals]
    -1533632,  # Plasma current [Amps]
    -3.231962138124,
)  # vacuum f = R*Bt

#########################################
# Coil current constraints
#
# Specify locations of the X-points
# to use to constrain coil currents

xpoints = [
    (1.285, -1.176),  # (R,Z) locations of X-points
    (1.2, 1.0),
]

isoflux = [(1.285, -1.176, 1.2, 1.2)]  # (R1,Z1, R2,Z2) pair of locations

constrain = freegs.control.constrain(xpoints=xpoints, gamma=1e-12, isoflux=isoflux)

constrain(eq)

#########################################
# Nonlinear solve

freegs.solve(
    eq,  # The equilibrium to adjust
    profiles,  # The plasma profiles
    constrain,  # Plasma control constraints
    show=True,
)  # Shows results at each nonlinear iteration

# eq now contains the solution

print("Done!")

print(f"Plasma current: {eq.plasmaCurrent():e} Amps")
print(f"Pressure on axis: {eq.pressure(0.0):e} Pascals")
print(f"Plasma poloidal beta: {eq.poloidalBeta():e}")
print(f"Plasma volume: {eq.plasmaVolume():e} m^3")

eq.tokamak.printCurrents()

# plot equilibrium
axis = eq.plot(show=False)
tokamak.plot(axis=axis, show=False)
constrain.plot(axis=axis, show=True)

# Safety factor
plt.plot(*eq.q())
plt.xlabel(r"Normalised $\psi$")
plt.ylabel("Safety factor")
plt.grid()
plt.show()

##############################################
# Save to geqdsk file

from freegs import geqdsk

with open("diiid.geqdsk", "w") as f:
    geqdsk.write(eq, f)
