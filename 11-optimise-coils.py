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

def weighted_sum(*args):
    """
    args should be pairs of functions and weights
    """
    return lambda eq: sum(func(eq) * weight for func, weight in args)

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
import copy
import bisect

def mutate(eq, controls):
    eq2 = copy.deepcopy(eq)
    for control in controls:
        control.set(eq2, control.get(eq2) * (1. + 0.1 * (random.random() - 0.5)))
    return eq2

def optimise(eq, profiles, constrain, controls, evaluate, N=10, CR=0.3, F=1.0, showall=False, show=False):
    """
    Use Differential Evolution to optimise an equilibrium
    https://en.wikipedia.org/wiki/Differential_evolution
    
    N >= 4 is the population size
    CR [0,1] is the crossover probability
    F  [0,2] is the differential weight

    showall  If true, show the iteration process for every solution
    show   If true, show best solution at each generation
    """
    assert N >= 4
    
    def measure(eq):
        # Need to update some internal caches
        eq._pgreen = eq.tokamak.createPsiGreens(eq.R, eq.Z)
        try:
            # Re-solve
            freegs.solve(eq,
                         profiles,
                         constrain, show=showall)
        except:
            # Solve failed.
            return float("inf")
        # Call user-supplied evaluation function
        return evaluate(eq)

    if show:
        import matplotlib.pyplot as plt
        from freegs.plotting import plotEquilibrium
        fig = plt.figure()
        axis = fig.add_subplot(111)
    
    best = (measure(eq), eq)  # Highest score, candidate solution (agent)
    population = [best]  # List of (score, agent)

    for i in range(N-1):
        agent = mutate(eq, controls)
        score_agent = (measure(agent), agent)
        population.append(score_agent)
        if score_agent[0] > best[0]:
            best = score_agent

    for generation in range(10):
        next_pop = [] # Next generation
        for ai, agent in enumerate(population):
            # Pick three other random agents, all different
            inds = [ai] # Sorted list of indices. Used to avoid clashes
            others = [] # The list of three agents
            for i in range(3):
                newind = random.randint(0, N-2-i)
                for ind in inds:
                    if newind == ind:
                        newind += 1
                bisect.insort(inds, newind) # Insert into sorted list
                others.append(population[newind][1])
            
            new_eq = copy.deepcopy(agent[1])
            R = random.randint(0, len(controls)-1) # Pick a random control to modify
            for i, control in enumerate(controls):
                if i == R or random.random() < CR:
                    control.set(new_eq,
                                control.get(others[0]) + F * (control.get(others[1]) - control.get(others[2])))
            score = measure(new_eq)
            if score < agent[0]:
                # Better than original
                new_agent = (score, new_eq)
                next_pop.append(new_agent)
                if score < best[0]:
                    # Now the best candidate
                    best = new_agent
            else:
                next_pop.append(agent)
        # end of generation. Print best score
        print("\nGeneration: {}, best score: {}\n".format(generation, best[0]))

        if show:
            axis.clear()
            plotEquilibrium(best[1], axis=axis, show=False)
            # Update the canvas and pause
            # Note, a short pause is needed to force drawing update
            axis.figure.canvas.draw()
            axis.set_title("Generation: {} Score: {}".format(generation, best[0]))
            fig.savefig("generation_{}.pdf".format(generation))
            plt.pause(0.5)
        population = next_pop
    # Finished, return best candidate
    return best[1]
            
# Currents in the coils
tokamak.printCurrents()

# Forces on the coils
eq.printForces()

best_eq = optimise(eq, profiles, constrain,
                   [coil_radius("P2U"),
                    coil_radius("P2L")],
                   max_coil_force, N=5, show=True)

# Forces on the coils
best_eq.printForces()
best_eq.plot()
