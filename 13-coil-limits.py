#!/usr/bin/env python

import freegs

#########################################
# Create the machine, which specifies coil locations
# and equilibrium, specifying the domain to solve over

tokamak = freegs.machine.TestTokamak()

eq = freegs.Equilibrium(tokamak=tokamak,
                        Rmin=0.1, Rmax=2.0,    # Radial domain
                        Zmin=-1.0, Zmax=1.0,   # Height range
                        nx=65, ny=65,          # Number of grid points
                        boundary=freegs.boundary.freeBoundaryHagenow)  # Boundary condition


#########################################
# Plasma profiles

profiles = freegs.jtor.ConstrainPaxisIp(eq,
                                        1e3, # Plasma pressure on axis [Pascals]
                                        2e5, # Plasma current [Amps]
                                        2.0) # Vacuum f=R*Bt

#########################################
# Coil current constraints
#
# Specify locations of the X-points
# to use to constrain coil currents

xpoints = [(1.1, -0.6),   # (R,Z) locations of X-points
           (1.1, 0.8)]

isoflux = [(1.1,-0.6, 1.1,0.6)] # (R1,Z1, R2,Z2) pair of locations

current_lims = [(-150000.0,140000.0),(0.0,65000.0),(-105000.0,0.0),(-60000.0,0.0)] # Lower/Upper limits of coil currents
total_current = 350000.0 # Limit on sum of absoloute circuit + standalone coil currents

constrain = freegs.control.constrain(xpoints=xpoints, isoflux=isoflux, current_lims=current_lims, max_total_current = total_current)

#########################################
# Nonlinear solve

freegs.solve(eq,          # The equilibrium to adjust
             profiles,    # The toroidal current profile function
             constrain,   # Constraint function to set coil currents
             show=True)

# eq now contains the solution

print("Done!")

print("Plasma current: %e Amps" % (eq.plasmaCurrent()))
print("Plasma pressure on axis: %e Pascals" % (eq.pressure(0.0)))
print("Poloidal beta: %e" % (eq.poloidalBeta()))

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
