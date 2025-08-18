"""
Routines for optimising equilibria
These make use of the generic "optimiser" routines

Copyright 2019 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk

This file is part of FreeGS.

FreeGS is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FreeGS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGS.  If not, see <http://www.gnu.org/licenses/>.

"""

import matplotlib.pyplot as plt
from numpy import inf, sqrt

from freegs.plotting import plotEquilibrium

from . import optimiser, picard

# Measures which operate on Equilibrium objects


def max_abs_coil_current(eq):
    """
    Given an equilibrium, return the maximum absolute coil current
    """
    currents = eq.tokamak.getCurrents()  # dictionary
    return max(abs(current) for current in currents.values())


def max_coil_force(eq):
    forces = eq.tokamak.getForces()  # dictionary
    return max(sqrt(force[0] ** 2 + force[1] ** 2) for force in forces.values())


def no_wall_intersection(eq):
    """Prevent intersection of LCFS with walls by returning inf if intersections are found"""
    if eq.intersectsWall():
        return float("inf")
    return 0.0  # No intersection


# Combine measures


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
            args_with_weights.append((arg, 1.0))
        else:
            args_with_weights.append(arg)

    def combined_measure(eq):
        return sum(func(eq) * weight for func, weight in args_with_weights)

    return combined_measure


# Controls for Equilibrium objects


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


# Monitor optimisation solutions


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
        self.axis.set_title(f"Generation: {generation} Score: {best[0]}")
        self.fig.savefig(f"generation_{generation}.pdf")
        plt.pause(0.5)


def optimise(eq, controls, measure, maxgen=10, N=10, CR=0.3, F=1.0, monitor=None):
    """Use Differential Evolution to optimise an Equilibrium
    https://en.wikipedia.org/wiki/Differential_evolution

    Parameters
    ----------
    eq:
        `Equilibrium` to be used as starting solution. These are deep
        copied, and passed as arguments to controls and measure
        functions
    controls:
        List of control objects. These objects must have methods:

        - ``.set(eq, value)`` which modifies the given `Equilibrium`
        - ``.get(eq)``  which returns the value from the `Equilibrium`

    measure(eq):
        Function which returns a score (value) for a given
        `Equilibrium`. The optimiser tries to minimise this value.
    maxgen:
        Maximum number of generations
    N:
        Population size (must be >= 4)
    CR:
        Crossover probability (must be in ``[0,1]``)
    F:
        Differential weight (must be in ``[0,2]``)
    monitor(generation, best, population):
        A function to be called each generation with the best
        Equilibrium and the whole population

        - generation = integer
        - best = (score, object)
        - population = [(score, object)]

    Returns
    -------
    ~.Equilibrium:
        The :class:`~.Equilibrium` with the lowest measure (score).

    """

    def solve_and_measure(eq):
        # Need to update some internal caches
        eq._pgreen = eq.tokamak.createPsiGreens(eq.R, eq.Z)
        try:
            # Re-solve
            picard.solve(eq, eq._profiles, eq._constraints)
            # Call user-supplied evaluation function
            return measure(eq)
        except:
            # Solve failed.
            return float(inf)

    # Call the generic optimiser,
    return optimiser.optimise(
        eq, controls, solve_and_measure, maxgen=maxgen, N=N, CR=CR, F=F, monitor=monitor
    )
