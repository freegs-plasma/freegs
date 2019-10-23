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

def max_abs_coil_current(eq):
    """
    Given an equilibrium, return the maximum absolute coil current
    """
    currents = eq.tokamak.getCurrents() # dictionary
    return max(abs(current) for current in currents.values())

def max_coil_force(eq):
    forces = eq.tokamak.getForces() # dictionary
    return max(sqrt(force[0]**2 + force[1]**2) for force in forces.values())

#### Scaling, limits

def scale(opt_func, value):
    return lambda eq: opt_func(eq) * value

def soft_upper_limit(opt_func, value, width=0.1):
    return lambda eq: 0.5*(1.0 + tanh((opt_func(eq)/value - 1.0)/width))


#### Controls

class coil_radius:
    def __init__(self, label):
        self.label = label
    
    def set(self, eq, R):
        eq.tokamak[self.label].R = R
        
    def get(self, eq):
        return eq.tokamak[self.label].R

############################

from scipy import optimize
import random

def optimise(eq, measures, controls):
    # Get initial values for all controls
    initial_values = [control.get(eq) for control in controls]

    def evaluate(values):
        # Modify settings
        for control, value in zip(controls, values):
            control.set(eq, value)
            
        print("SETTING: ", values)

        # Need to update some internal caches
        eq._pgreen = eq.tokamak.createPsiGreens(eq.R, eq.Z)
        
        # Re-solve
        freegs.solve(eq,
                     profiles,
                     constrain, show=True)

        # Sum measures to get overall score
        return sum(measure(eq) for measure in measures)

    best_values = initial_values
    best_measure = evaluate(best_values)
    for i in range(10):
        new_values = [value * (1. + 0.1 * (random.random() - 0.5)) for value in best_values]
        measure = evaluate(new_values)
        if (measure < best_measure):
            best_values = new_values
            best_measure = measure
        print("MEASURE: ", best_measure)
    
    return best_values
    #return optimize.minimize(evaluate, initial_values)

# Currents in the coils
tokamak.printCurrents()

# Forces on the coils
eq.printForces()

print(optimise(eq, [max_coil_force], [coil_radius("P2U")]))


