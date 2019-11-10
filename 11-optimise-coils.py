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

profiles = freegs.jtor.ConstrainPaxisIp(1e3, # Plasma pressure on axis [Pascals]
                                        2e5, # Plasma current [Amps]
                                        2.0) # Vacuum f=R*Bt

#########################################
# Coil current constraints
#
# Specify locations of the X-points
# to use to constrain coil currents

xpoints = [(1.1, -0.6),   # (R,Z) locations of X-points
           (1.1, 0.6)]

isoflux = [(1.1,-0.6, 1.1,0.6), # (R1,Z1, R2,Z2) pair of locations
           (1.7, 0.0, 0.84, 0.0)]

constrain = freegs.control.constrain(xpoints=xpoints, isoflux=isoflux, gamma = 1e-17)

#########################################
# Nonlinear solve

freegs.solve(eq,          # The equilibrium to adjust
             profiles,    # The toroidal current profile function
             constrain)   # Constraint function to set coil currents


            
# Currents in the coils
tokamak.printCurrents()

# Forces on the coils
eq.printForces()

############################
# Optimise
# Minimise the maximum force on the coils, while avoiding intersection of the LCFS and walls
# by modifying the radius of the P2U and P2L coils.

from freegs import optimise as opt

best_eq = opt.optimise(eq,  # Starting equilibrium
                       # List of controls
                       [opt.CoilRadius("P2U"),
                        opt.CoilRadius("P2L"), opt.CoilHeight("P2L")],
                       # The function to minimise
                       opt.weighted_sum(opt.max_coil_force, opt.no_wall_intersection),
                       N=10,  # Number of solutions in each generation
                       maxgen=20, # How many generations
                       monitor=opt.PlotMonitor())  # Plot the best in each generation

# Forces on the coils
best_eq.printForces()
best_eq.plot()
