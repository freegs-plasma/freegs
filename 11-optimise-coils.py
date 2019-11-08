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

#### Optimisation functions

from math import sqrt

from freegs import polygons

def max_abs_coil_current(eq):
    """
    Given an equilibrium, return the maximum absolute coil current
    """
    currents = eq.tokamak.getCurrents() # dictionary
    return max(abs(current) for current in currents.values())

def max_coil_force(eq):
    forces = eq.tokamak.getForces() # dictionary
    return max(sqrt(force[0]**2 + force[1]**2) for force in forces.values())

def no_wall_intersection(eq):
    """Prevent intersection of LCFS with walls by returning inf if intersections are found"""
    separatrix = eq.separatrix() # Array [:,2]
    wall = eq.tokamak.wall # Wall object with R and Z members (lists)

    if polygons.intersect(separatrix[:,0], separatrix[:,1],
                          wall.R, wall.Z):
        return float("inf")
    return 0.0 # No intersection


#### Scaling, limits

def scale(opt_func, value):
    return lambda eq: opt_func(eq) * value

def soft_upper_limit(opt_func, value, width=0.1):
    return lambda eq: 0.5*(1.0 + tanh((opt_func(eq)/value - 1.0)/width))

def weighted_sum(*args):
    """
    Returns a function which takes a single argument (the equilibrium),
    and passes it to a set of given functions. These functions are assumed
    to return values, which are multiplied by weights, summed and returned. 
    
    args should be either functions or pairs of functions and weights.
    If no weights are supplied then a weight of 1.0 is assumed.
    """
    args_with_weights = []
    for arg in args:
        if callable(arg):
            args_with_weights.append( (arg, 1.0) )
        else:
            args_with_weights.append( arg )
    
    return lambda eq: sum(func(eq) * weight for func, weight in args_with_weights)

#### Controls

class coil_radius:
    def __init__(self, label):
        self.label = label
    
    def set(self, eq, R):
        eq.tokamak[self.label].R = R
        
    def get(self, eq):
        return eq.tokamak[self.label].R

############################

def make_eq_measure(profiles, constrain, evaluate):
    def measure(eq):
        # Need to update some internal caches
        eq._pgreen = eq.tokamak.createPsiGreens(eq.R, eq.Z)
        try:
            # Re-solve
            freegs.solve(eq,
                         profiles,
                         constrain)
        except:
            # Solve failed.
            return float("inf")
        # Call user-supplied evaluation function
        return evaluate(eq)
    return measure

import matplotlib.pyplot as plt
from freegs.plotting import plotEquilibrium

# Plot and save the best equilibrium each generation
class PlotMonitor:
    def __init__(self):
        self.fig = plt.figure()
        self.axis = self.fig.add_subplot(111)

    def __call__(self, generation, best, population):
        self.axis.clear()
        plotEquilibrium(best[1], axis=self.axis, show=False)
        # Update the canvas and pause
        # Note, a short pause is needed to force drawing update
        self.fig.canvas.draw()
        self.axis.set_title("Generation: {} Score: {}".format(generation, best[0]))
        self.fig.savefig("generation_{}.pdf".format(generation))
        plt.pause(0.5)
            
# Currents in the coils
tokamak.printCurrents()

# Forces on the coils
eq.printForces()

from freegs import optimiser

best_eq = optimiser.optimise(eq,
                             [coil_radius("P2U"),
                              coil_radius("P2L")],
                             make_eq_measure(profiles, constrain,
                                             weighted_sum(max_coil_force, no_wall_intersection)),
                             N=10, monitor=PlotMonitor(), maxgen=20)

# Forces on the coils
best_eq.printForces()
best_eq.plot()
