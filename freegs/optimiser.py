""" 
Differential Evolution optimisation

Implemented generically, to optimise opaque objects

The optimiser copies objects using deepcopy, and manipulates objects
using supplied control objects.

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

import random
import copy
import bisect

def mutate(obj, controls):
    """
    Create a new object by taking an object and a set of control objects
    to change randomly

    obj        An object to copy. This will be deepcopied, not modified
    controls   A list of control objects
    """
    new_obj = copy.deepcopy(obj)
    for control in controls:
        # Add 10% variations to controls
        new_value = control.get(obj) * (1. + 0.1 * (random.random() - 0.5))
        if abs(new_value) < 1e-10:
            # Don't know what scale, so add ~1 random value to it
            new_value += random.normal(loc=0.0, scale=1.0)
        control.set(new_obj, new_value)
    return new_obj

def pickUnique(N, m, e):
    """Pick m random values from the range 0...(N-1), excluding those in list e
    The returned values should be in a random order (not sorted)
    """
    assert m <= N - len(e)
    
    inds = sorted(e) # Sorted list of indices. Used to avoid clashes
    others = [] # The list of three agents
    for i in range(m):
        newind = random.randint(0, N-1-i-len(e))
        for ind in inds:
            if newind == ind:
                newind += 1
        bisect.insort(inds, newind)
        others.append(newind)
    return others

def optimise(obj, controls, measure, maxgen=10, N=10, CR=0.3, F=1.0, monitor=None):
    """Use Differential Evolution to optimise an object
    https://en.wikipedia.org/wiki/Differential_evolution
    
    obj  is an object to be used as starting solution
         These objects are deep copied, and passed as arguments
         to controls and measure functions
    
    controls   A list of control objects. These objects must
               have methods:
         .set(object, value) which modifies the given object
         .get(object)  which returns the value from the object

    measure(object) is a function which returns a score (value) for a
                    given object. The optimiser tries to minimise this
                    value.
    
    maxgen   is the maximum number of generations

    N >= 4 is the population size
    CR [0,1] is the crossover probability
    F  [0,2] is the differential weight
    
    monitor(generation, best, population) 

           A function to be called each generation with the best
           solution and the whole population 
           generation = integer
           best = (score, object)
           population = [(score, object)]

    Returns the object with the lowest measure (score).

    """
    assert N >= 4
    
    best = (measure(obj), obj)  # Highest score, candidate solution (agent)
    population = [best]  # List of (score, agent)

    for i in range(N-1):
        agent = mutate(obj, controls)
        score_agent = (measure(agent), agent)
        population.append(score_agent)
        if score_agent[0] > best[0]:
            best = score_agent

    for generation in range(maxgen):
        next_pop = [] # Next generation
        for ai, agent in enumerate(population):
            # Pick three other random agents, all different
            others = [population[index][1] for index in pickUnique(N, 3, [ai])]
            
            new_obj = copy.deepcopy(agent[1])
            R = random.randint(0, len(controls)-1) # Pick a random control to modify
            for i, control in enumerate(controls):
                if i == R or random.random() < CR:
                    control.set(new_obj,
                                control.get(others[0]) + F * (control.get(others[1]) - control.get(others[2])))
            score = measure(new_obj)
            if score < agent[0]:
                # Better than original
                new_agent = (score, new_obj)
                next_pop.append(new_agent)
                if score < best[0]:
                    # Now the best candidate
                    best = new_agent
            else:
                next_pop.append(agent)
        # End of generation. Call monitor with best object, score
        if monitor:
            monitor(generation, best, next_pop)
        # Change to next generation
        population = next_pop
    # Finished, return best candidate
    return best[1]
