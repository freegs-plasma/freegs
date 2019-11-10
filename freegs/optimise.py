#
# Routines for optimising equilibria
# These make use of the generic "optimiser" routines
#

from . import optimiser
from . import polygons
from . import picard

from math import sqrt

### Measures which operate on Equilibrium objects

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


### Combine measures

def weighted_sum(*args):
    """
    Combine measures together
    
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

#### Controls for Equilibrium objects

class CoilRadius:
    """A control to modify the radius of a specified coil"""
    def __init__(self, label, minimum=0.0, maximum=None):
        """
        label  : string   The label of a coil to be modified
        minimum : number, optional   The minimum allowed radius
        maximum : number, optional   The maximum allowed radius
        """
        self.label = label
        self.minimum = minimum
        self.maximum = maximum
    
    def set(self, eq, R):
        if self.minimum and (R < self.minimum):
            R = self.minimum
        if self.maximum and (R > self.maximum):
            R = self.maximum
        eq.tokamak[self.label].R = R
        
    def get(self, eq):
        return eq.tokamak[self.label].R

class CoilHeight:
    """A control to modify the height of a specified coil"""
    def __init__(self, label, minimum=None, maximum=None):
        """
        label  : string   The label of a coil to be modified
        minimum : number, optional   The minimum allowed height
        maximum : number, optional   The maximum allowed height
        """
        self.label = label
        self.minimum = minimum
        self.maximum = maximum
    
    def set(self, eq, Z):
        if self.minimum and (Z < self.minimum):
            Z = self.minimum
        if self.maximum and (Z > self.maximum):
            Z = self.maximum
        eq.tokamak[self.label].Z = Z
        
    def get(self, eq):
        return eq.tokamak[self.label].Z
    
import matplotlib.pyplot as plt
from freegs.plotting import plotEquilibrium

#### Monitor optimisation solutions

# Plot and save the best equilibrium each generation
class PlotMonitor:
    """
    Plot the best solution at the end of each generation,
    saves the plot to a PNG file. 
    """
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

def optimise(eq, controls, measure, maxgen=10, N=10,
             CR=0.3, F=1.0, monitor=None):
    """Use Differential Evolution to optimise an Equilibrium
    https://en.wikipedia.org/wiki/Differential_evolution
    
    eq   is an Equilibrium to be used as starting solution
         These are deep copied, and passed as arguments
         to controls and measure functions
    
    controls   A list of control objects. These objects must
               have methods:
         .set(eq, value) which modifies the given Equilibrium
         .get(eq)  which returns the value from the Equilibrium

    measure(eq) is a function which returns a score (value) for a
                given Equilibrium. The optimiser tries to minimise this
                value.
    
    maxgen   is the maximum number of generations

    N >= 4 is the population size
    CR [0,1] is the crossover probability
    F  [0,2] is the differential weight
    
    monitor(generation, best, population) 

           A function to be called each generation with the best
           Equilibrium and the whole population 
           generation = integer
           best = (score, object)
           population = [(score, object)]

    Returns the Equilibrium with the lowest measure (score).

    """
    
    def solve_and_measure(eq):
        # Need to update some internal caches
        eq._pgreen = eq.tokamak.createPsiGreens(eq.R, eq.Z)
        try:
            # Re-solve
            picard.solve(eq,
                         eq._profiles,
                         eq._constraints)
        except:
            # Solve failed.
            return float("inf")
        # Call user-supplied evaluation function
        return measure(eq)

    # Call the generic optimiser, 
    return optimiser.optimise(eq, controls, solve_and_measure, maxgen=maxgen, N=N,
                              CR=CR, F=F, monitor=monitor)
