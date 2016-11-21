#
# Performs the same calculation with two different boundary locations
# and compares the result


import freegs

from numpy import linspace, amin, amax, meshgrid, exp

import matplotlib.pyplot as plt

tokamak = freegs.machine.TestTokamak()

eq1 = freegs.Equilibrium(tokamak=tokamak,
                         Rmin=0.1, Rmax=2.0,    # Radial domain
                         Zmin = -2.0, Zmax = 2.0,
                         nx=65, ny=129)          # Number of grid points

tokamak = freegs.machine.TestTokamak()

eq2 = freegs.Equilibrium(tokamak=tokamak,
                         Rmin=0.1, Rmax=2.0,    # Radial domain
                         Zmin=-1.0, Zmax=1.0,   # Height range
                         nx=65, ny=65)          # Number of grid points

# Set the initial psi

xx, yy = meshgrid(linspace(eq1.Rmin,eq1.Rmax,65), linspace(eq1.Zmin,eq1.Zmax,129), indexing='ij')
psi = exp( - ( (xx - 1.0)**2 + (yy)**2 ) / 0.3**2 )
eq1._updatePlasmaPsi(psi)

xx, yy = meshgrid(linspace(eq2.Rmin,eq2.Rmax,65), linspace(eq2.Zmin,eq2.Zmax,65), indexing='ij')
psi = exp( - ( (xx - 1.0)**2 + (yy)**2 ) / 0.3**2 )
eq2._updatePlasmaPsi(psi)


xpoints = [(1.1, -0.6),   # (R,Z) locations of X-points
           (1.1, 0.8)]

isoflux = [(1.1,-0.6, 1.1,0.6)]

constrain = freegs.control.constrain(xpoints=xpoints, isoflux=isoflux)

constrain(eq1)
constrain(eq2)

profiles = freegs.jtor.ConstrainBetapIp(0.1, 1e6, 1.0)

jtor1 = profiles.Jtor(eq1.R, eq1.Z, eq1.psi())
jtor2 = profiles.Jtor(eq2.R, eq2.Z, eq2.psi())

# Check jtor1 == jtor2

# Check a single linear solve
eq1.solve(profiles)
eq2.solve(profiles)

# Nonlinear solve
#freegs.solve(eq1, profiles, constrain)
#freegs.solve(eq2, profiles, constrain)

psi1 = eq1.psi()
psi2 = eq2.psi()

#psi1 = eq1.tokamak.psi(eq1.R, eq1.Z)
#psi2 = eq2.tokamak.psi(eq2.R, eq2.Z)

#psi1 = eq1.plasma_psi
#psi2 = eq2.plasma_psi

levels = linspace(amin(psi1), amax(psi1), 40)

plt.contour(eq1.R, eq1.Z, psi1, levels=levels, colors='k')
plt.contour(eq2.R, eq2.Z, psi2, levels=levels, colors='r')
plt.show()




