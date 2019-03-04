#!/usr/bin/env python

import freegs
from freegs.plotting import plotConstraints

#########################################
# Create the machine, which specifies coil locations
# and equilibrium, specifying the domain to solve over

tokamak = freegs.machine.MASTU()


eq = freegs.Equilibrium(tokamak=tokamak,
                        Rmin=0.1, Rmax=2.0,    # Radial domain
                        Zmin=-2.1, Zmax=2.1,   # Height range
                        nx=65, ny=65)          # Number of grid points

#########################################
# Plasma profiles

profiles = freegs.jtor.ConstrainPaxisIp(2e4, # Plasma pressure on axis [Pascals]
                                        6e5, # Plasma current [Amps]
                                        1.0) # vacuum f = R*Bt

#########################################
# Coil current constraints
#
# Specify locations of the X-points
# to use to constrain coil currents

Rx = 0.509
Zx = 1.291

Rmid = 1.34   # Outboard midplane
Rin = 0.3581  # Inboard midplane

xpoints = [(Rx, -Zx),   # (R,Z) locations of X-points
           (Rx,  Zx)]

isoflux = [(Rx,-Zx, Rmid, 0.0)   # Outboard midplane, lower X-point
           ,(Rx,Zx, Rmid, 0.0)   # Outboard midplane, upper X-point

           # Link inner and outer midplane locations
           ,(Rmid, 0.0, Rin, 0.0)

           # Separatrix in the divertor chamber
           ,(Rx,-Zx, 0.95, -1.77)
           ,(Rx, Zx, 0.95,  1.77)
           ]

constrain = freegs.control.constrain(xpoints=xpoints, gamma=1e-5, isoflux=isoflux)

constrain(eq)

#########################################
# Nonlinear solve

freegs.solve(eq,          # The equilibrium to adjust
             profiles,    # The plasma profiles
             constrain,   # Plasma control constraints
             show=False)   # Shows results at each nonlinear iteration

#########################################
# Now adjust the equilibrium manually
# 

isoflux = [(Rx,-Zx, Rmid, 0.0)   # Outboard midplane, lower X-point
           ,(Rx,Zx, Rmid, 0.0)   # Outboard midplane, upper X-point

           ,(Rmid, 0.0, Rin, 0.0)
           
           ,(Rx,-Zx, 0.95, -1.77)
           ,(Rx, Zx, 0.95,  1.77)

           ,(Rx,-Zx, 0.76, -1.58)
           ,(Rx, Zx, 0.76,  1.58)
           
           ,(Rx,-Zx, 1.25, -1.8)
           ,(Rx, Zx, 1.25,  1.8)

           ]

constrain = freegs.control.constrain(xpoints=xpoints, gamma=1e-12, isoflux=isoflux)

# Turn off feedback control for all coils
for label, coil in tokamak.coils:
    coil.control = False

# Centre column coil
tokamak["Pc"].current = -3e4
    
# Turn on vertical feedback control
tokamak["P61"].control = True

# Coil in the "nose" of the divertor
tokamak["Dp"].current = -1000.0

# At top of divertor chamber
tokamak["D6"].current = -500.0

# X-point location
tokamak["Px"].current = 2000.0

tokamak["D1"].current = 1000.0

# Coil in outer corner 
tokamak["D5"].current = 1500.

# Coil at bottom centre
tokamak["D3"].current = 2800

freegs.solve(eq,          # The equilibrium to adjust
             profiles,    # The plasma profiles
             constrain,   # Plasma control constraints
             show=True)   # Shows results at each nonlinear iteration

# eq now contains the solution

print("Done!")

print("Plasma current: %e Amps" % (eq.plasmaCurrent()))
print("Pressure on axis: %e Pascals" % (eq.pressure(0.0)))
print("Poloidal beta: %e" % (eq.poloidalBeta()))
print("Plasma volume: %e m^3" % (eq.plasmaVolume()))

eq.tokamak.printCurrents()

axis = eq.plot(show=False)
constrain.plot(axis=axis)

##############################################
# Save to geqdsk file

from freegs import geqdsk

with open("mast-upgrade.geqdsk", "w") as f:
    geqdsk.write(eq, f)

# Call matplotlib show so plot pauses
import matplotlib.pyplot as plt
plt.show()
