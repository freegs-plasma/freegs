from freegs import geqdsk
from freegs import machine
from freegs.plotting import plotEquilibrium

'''
Solution from 01-freeboundary.py

P1L : ShapedCoil([(0.95, -1.15), (0.95, -1.05), (1.05, -1.05), (1.05, -1.15)], current=153998.7, turns=1, control=True)
P1U : ShapedCoil([(0.95, 1.15), (0.95, 1.05), (1.05, 1.05), (1.05, 1.15)], current=62444.5, turns=1, control=True)
P2L : Coil(R=1.75, Z=-0.6, current=-99080.9, turns=1, control=True)
P2U : Coil(R=1.75, Z=0.6, current=-56656.4, turns=1, control=True)

'''

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

constrain = freegs.control.constrain(xpoints=xpoints, isoflux=isoflux)

#########################################
# Nonlinear solve

freegs.solve(eq,          # The equilibrium to adjust
             profiles,    # The toroidal current profile function
             constrain,
             show=True)   # Constraint function to set coil currents

# eq now contains the solution

# Currents in the coils
tokamak.printCurrents()

psi1 = eq.psi()

#########################################

# Save to G-EQDSK file

from freegs import geqdsk

with open("../lsn.geqdsk", "w+") as f:
    geqdsk.write(eq, f)

#########################################
# Read in the geqdsk file with bounds on the coil currents

current_lims = [(140000.0,155000.0),(60000.0,63000.0),(-105000.0,-90000.0),(-60000.0,-55000.0)] # Lower/Upper limits of coil currents
#current_lims = [(-150000.0,155000.0),(0.0,65000.0),(-105000.0,0.0),(-60000.0,0.0)] # Lower/Upper limits of coil currents
#current_lims = [(140000.0,150000.0),(60000.0,65000.0),(-105000.0,-90000.0),(-60000.0,-55000.0)] # Lower/Upper limits of coil currents

total_current = 350000.0 # Limit on sum of absoloute circuit + standalone coil currents

constraints = {
    'current_bounds':current_lims,
    'total_current':total_current,
}

tokamak2 = freegs.machine.TestTokamak()

with open("../lsn.geqdsk") as f:
    eq2= geqdsk.read(f, tokamak2, show=True, coil_constraints=constraints)

plotEquilibrium(eq2)
tokamak2.printCurrents()

psi2 = eq2.psi()

pct_change = abs(100.0*(psi2 - psi1)/psi1)

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.imshow(pct_change.T,extent=[min(eq.R[:, 0]),max(eq.R[:, 0]),min(eq.Z[0, :]),max(eq.Z[0, :])],origin='lower',vmax=20.0)
ax.set_xlabel('R(m)')
ax.set_ylabel('Z(m)')
ax.set_aspect('equal')
ax.set_title('Max pct diff: '+str(np.max(pct_change)))
plt.show()