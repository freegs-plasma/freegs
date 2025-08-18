#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import freegs
from freegs import critical, geqdsk
from freegs.plotting import plotEquilibrium

"""
First creates a limited plasma and solves, before
creating a diverted plasma (which will have a larger CSA)
and solves under identical constraints. Due to the reduced CSA
of the limited plasma, one expects the plasma profiles to have a larger
magnitude when limited such that J is larger. J is expected to be larger
such that when J is integrated over a smaller CSA, the resultant Ip (and BetaP)
are the same. Consequently, different coil currents are also expected.
"""

#########################################
# Create the machine, which specifies coil locations
# and equilibrium, specifying the domain to solve over

tokamak = freegs.machine.TestTokamakLimited()

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

profiles = freegs.jtor.ConstrainBetapIp(
    eq,
    0.15,  # Plasma poloidal beta
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

isoflux = [
    (1.1, -0.6, 1.1, 0.6),
    (1.1, -0.6, 0.732, 0.0426),
]  # (R1,Z1, R2,Z2) pair of locations

constrain = freegs.control.constrain(xpoints=xpoints, isoflux=isoflux)

#########################################
# Nonlinear solve

freegs.solve(eq, profiles, constrain, show=True, check_limited=True, limit_it=0)

# eq now contains the solution

print("Done!")

print("Plasma current: %e Amps" % (eq.plasmaCurrent()))
print("Plasma pressure on axis: %e Pascals" % (eq.pressure(0.0)))
print("Poloidal beta: %e" % (eq.poloidalBeta()))

# Currents in the coils
tokamak.printCurrents()

##############################################
# Final plot of equilibrium

axis = eq.plot(show=False)
eq.tokamak.plot(axis=axis, show=False)
constrain.plot(axis=axis, show=True)

###############################################
# Check that the core is masked correctly

fig, ax = plt.subplots()

ax.contour(eq.R, eq.Z, eq.psiN(), levels=30, colors="b")
ax.contour(eq.R, eq.Z, eq.psiN(), levels=[1.0], colors="orange")
ax.plot(eq.tokamak.wall.R, eq.tokamak.wall.Z, color="k")
opt, xpt = critical.find_critical(eq.R, eq.Z, eq.psi())
isoflux = np.array(
    freegs.critical.find_separatrix(
        eq, ntheta=101, opoint=opt, xpoint=xpt, psi=eq.psi()
    )
)
ind = np.argmin(isoflux[:, 1])
rbdry = np.roll(isoflux[:, 0][::-1], -ind)
rbdry = np.append(rbdry, rbdry[0])
zbdry = np.roll(isoflux[:, 1][::-1], -ind)
zbdry = np.append(zbdry, zbdry[0])
ax.plot(rbdry, zbdry, "rx")
ax.contourf(eq.R, eq.Z, eq.mask)
ax.set_aspect("equal")
ax.set_xlabel("R(m)")
ax.set_ylabel("Z(m)")
plt.show()


with open("limited.geqdsk", "w") as f:
    geqdsk.write(eq, f)

#########################################
# Create the machine, which specifies coil locations
# and equilibrium, specifying the domain to solve over

tokamak = freegs.machine.TestTokamak()

eq2 = freegs.Equilibrium(
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

profiles = freegs.jtor.ConstrainBetapIp(
    eq2,
    0.15,  # Plasma poloidal beta
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

isoflux = [
    (1.1, -0.6, 1.1, 0.6),
    (1.1, -0.6, 0.732, 0.0426),
]  # (R1,Z1, R2,Z2) pair of locations

# current_lims = [(-150000.0,140000.0),(0.0,65000.0),(-105000.0,0.0),(-60000.0,0.0)]
# total_current = 350000.0

constrain = freegs.control.constrain(
    xpoints=xpoints, isoflux=isoflux
)  # , current_lims=current_lims, max_total_current = total_current)

#########################################
# Nonlinear solve

freegs.solve(eq2, profiles, constrain, show=True, check_limited=True)

# eq now contains the solution

print("Done!")

print("Plasma current: %e Amps" % (eq2.plasmaCurrent()))
print("Plasma pressure on axis: %e Pascals" % (eq2.pressure(0.0)))
print("Poloidal beta: %e" % (eq2.poloidalBeta()))

# Currents in the coils
tokamak.printCurrents()

##############################################
# Final plot of equilibrium

axis = eq2.plot(show=False)
eq2.tokamak.plot(axis=axis, show=False)
constrain.plot(axis=axis, show=True)

plt.show()

###############################################
# Check that the core is masked correctly

fig, ax = plt.subplots()

ax.contour(eq2.R, eq2.Z, eq2.psiN(), levels=30, colors="b")
ax.contour(eq2.R, eq2.Z, eq2.psiN(), levels=[1.0], colors="orange")
ax.plot(eq2.tokamak.wall.R, eq2.tokamak.wall.Z, color="k")
opt, xpt = critical.find_critical(eq2.R, eq2.Z, eq2.psi())
mask = critical.core_mask(eq2.R, eq2.Z, eq2.psi(), opt, xpt, eq2.psi_bndry)
ax.contourf(eq2.R, eq2.Z, mask)
ax.set_aspect("equal")
ax.set_xlabel("R(m)")
ax.set_ylabel("Z(m)")
plt.show()
#################

# Compare plasma profiles between limited and diverted plasmas
psi_levels = np.linspace(0.0, 1.0, 100, endpoint=True)

fig, ax = plt.subplots()

ax.plot(psi_levels, eq.pprime(psi_levels), color="r", label="limited")
ax.plot(psi_levels, eq2.pprime(psi_levels), color="b", label="diverted")
ax.set_xlabel("psiN")
ax.set_ylabel("pprime")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(psi_levels, eq.pressure(psi_levels), color="r", label="limited")
ax.plot(psi_levels, eq2.pressure(psi_levels), color="b", label="diverted")
ax.set_xlabel("psiN")
ax.set_ylabel("p")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(psi_levels, eq.ffprime(psi_levels), color="r", label="limited")
ax.plot(psi_levels, eq2.ffprime(psi_levels), color="b", label="diverted")
ax.set_xlabel("psiN")
ax.set_ylabel("ffprime")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(psi_levels, eq.fpol(psi_levels), color="r", label="limited")
ax.plot(psi_levels, eq2.fpol(psi_levels), color="b", label="diverted")
ax.set_xlabel("psiN")
ax.set_ylabel("fpol")
ax.legend()
plt.show()

#########################
# Load in the geqdsk of the limited plasma
tokamak = freegs.machine.TestTokamakLimited()

with open("limited.geqdsk") as f:
    eq3 = geqdsk.read(f, tokamak, show=True)

# Plot equilibrium
plotEquilibrium(eq3)
