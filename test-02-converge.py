#!/usr/bin/env python
#
# Check convergence rate of a free boundary solution
# 

import freegs

import matplotlib.pyplot as plt

import pickle

xpoints = [(1.1, -0.6),   # (R,Z) locations of X-points
           (1.1, 0.8)]
isoflux = [(1.1,-0.6, 1.1,0.6)]

psiloc = (1.2, 0.1) # (R,Z) location to record psi

constrain = freegs.control.constrain(xpoints=xpoints, isoflux=isoflux, gamma=0.0)

# Constrain plasma pressure on axis and plasma current
#profiles = freegs.jtor.ConstrainPaxisIp(1e4, # Plasma pressure on axis [Pascals]
#                                        1e6, # Plasma current [Amps]
#                                        1.0) # Vacuum f=R*Bt


# Constrain poloidal beta and plasma current
profiles = freegs.jtor.ConstrainBetapIp(0.1, 1e6, 1.0)

resolutions = [33, 65, 129]#, 257, 513, 1025]

boundaries = [("A", 0.1, 2.0, -1.0, 1.0)
              ,("B", 0.5, 1.75, -0.8, 1.1)
          ]

for n in resolutions:
    for bndry_name, Rmin, Rmax, Zmin, Zmax in boundaries:
        # Re-create objects so no state is retained between runs
    
        tokamak = freegs.machine.TestTokamak()
        
        eq = freegs.Equilibrium(tokamak=tokamak,
                                Rmin=Rmin, Rmax=Rmax,    # Radial domain
                                Zmin=Zmin, Zmax=Zmax,   # Height range
                                nx=n, ny=n)          # Number of grid points

        freegs.solve(eq,          # The equilibrium to adjust
                     profiles,    # The toroidal current profile function
                     constrain,   # Constraint function to set coil currents
                     rtol = 1e-6, show=False)
        
        # Save solution for later analysis
        with open("test-02-"+bndry_name+"-"+str(n)+".pkl", "wb") as f:
            pickle.dump(n, f)
            pickle.dump( (bndry_name, Rmin, Rmax, Zmin, Zmax), f )
            pickle.dump( eq, f )
        
        



