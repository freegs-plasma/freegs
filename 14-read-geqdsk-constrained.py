"""
Creates an equilibrium before saving it as a geqdsk file.
Then reconstructs the equilibrium from the geqdsk using
constraints on coil currents.
"""

import freegs
from freegs import geqdsk
from freegs.plotting import plotEquilibrium

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

# Currents in the coils
tokamak.printCurrents()

psi1 = eq.psi()

#########################################

# Save to G-EQDSK file


with open("lsn.geqdsk", "w+") as f:
    geqdsk.write(eq, f)

#########################################
# Read in the geqdsk file with bounds on the coil currents

# Lower/Upper limits of coil currents
current_lims = [
    (140000.0, 155000.0),
    (60000.0, 63000.0),
    (-105000.0, -90000.0),
    (-60000.0, -55000.0),
]

tokamak2 = freegs.machine.TestTokamak()

with open("lsn.geqdsk") as f:
    eq2 = geqdsk.read(f, tokamak2, show=True, current_bounds=current_lims)

# eq2 now contains the solution

plotEquilibrium(eq2)

# Currents in the coils
tokamak2.printCurrents()

# Compare reconstructed psi v original psi
psi2 = eq2.psi()

pct_change = abs(100.0 * (psi2 - psi1) / psi1)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.imshow(
    pct_change.T,
    extent=[min(eq.R[:, 0]), max(eq.R[:, 0]), min(eq.Z[0, :]), max(eq.Z[0, :])],
    origin="lower",
    vmax=20.0,
)

ax.plot(eq.tokamak.wall.R, eq.tokamak.wall.Z, "k")
ax.contour(eq.R, eq.Z, eq.psi(), levels=[eq.psi_bndry], colors="w")
ax.contour(eq2.R, eq2.Z, eq2.psi(), levels=[eq2.psi_bndry], colors="r")
ax.plot([], [], "r", label="original")
ax.plot([], [], "w", label="achieved")

ax.set_xlabel("R(m)")
ax.set_ylabel("Z(m)")
ax.set_aspect("equal")
ax.set_title(r"pct diff in $\psi$")
ax.legend()

plt.show()
