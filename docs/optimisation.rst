.. _optimisation

Optimisation
============

This is an experimental feature which is at an early stage of development.  The
aim is to enable equilibria to be automatically optimised. This has the
following components:

#. Measures, a quantity which measures how "good" a solution is. Typically the
   aim is to minimise this quantity, so I suppose it's really a measure of how
   bad the solution is.
#. Controls, quantities which can be changed. These could be machine parameters
   such as coil locations, constraints like X-point location, or plasma profiles
   such as poloidal beta or plasma current.
#. An algorithm which modifies the controls and finds the best equilibrium
   according to the measure it's given. At the moment the method used is
   Differential Evolution.


Differential Evolution
----------------------

`Differential Evolution <https://en.wikipedia.org/wiki/Differential_evolution>`_ is a type
of stochastic search, similar to Genetic Algorithms, generally well suited to problems
involving continuously varying parameters.

The implementation of the algorithm is in ``freegs.optimiser``. It is generic,
in that it operates on objects but does not need to know any details of what
those objects are. To modify objects a list of ``controls`` are passed to the
optimiser, each of which can set and get a value.  To score each object a
``measure`` function is needed, which takes an object as input and returns a
value. The optimiser works to minimise this value.

An example which uses the optimisation method is in the ``freegs`` directory.
This optimises a quadratic in 2D rather than tokamak equilibria. 100 generations
are run, with 10 solutions (sometimes called agents) in each generation.  Run
this example with the command:

::

   python test_optimiser.py

This should produce the figure below. The red point is the best solution at each
generation; black points are the other points in that generation. Faded colors
(light red, grey) are used to show previous generations. It can be seen that the
points are clustered around the starting solution, as the agents spread out, and
then around the solution as the agents converge to the minimum.

.. image:: optimiser.gif
   :alt: Optimisation of a quadratic. The minimum is at (1,2), and starting point is at (0.1,0.1). Points mark the solutions tested at all generations.

         
