Grad-Shafranov solver
=====================

This solves the Grad-Shafranov equation for axisymmetric plasma equilibria, 
mainly for free boundary calculations.

Examples
--------

$ python example-fixedboundary.py 

This solves a fixed boundary problem, in which the square edges of the domain
are fixed. The plasma pressure on axis and plasma current are fixed.

$ python example-freeboundary.py

This solves a free boundary problem, specifying the desired location of two X-points

$ python example-xpoints.py

This demonstrates the coil current constraint code, finding X-points, and marking core region
These routines are then used inside the free boundary solver

Files
-----

boundary.py        - Operators for applying boundary conditions to plasma psi
constraints.py     - Routines for applying constraints to the plasma boundary
equilibrium.py     - Represents the plasma equilibrium state
gradshafranov.py   - Greens functions and operators for the Grad-Shafranov equation
jtor.py            - Routines for calculating toroidal current density
machine.py         - Represents the coils and power supply circuits
multigrid.py       - The multigrid solver for the linear elliptic operator
plotting.py        - Plotting routines using matplotlib
picard.py          - Nonlinear solver, iterating the profiles and constraints

Multigrid solver
----------------

The code in multigrid.py solves elliptic equations in a square domain, with Dirichlet (fixed value)
boundary conditions. The smoother is a simple Jacobi method, and 2nd order
central differences is used to discretise the operator. 
The test case can be run with:

    $ python multigrid.py

This runs two solves. Both start with an initial maximum residual of 1.0.
The first simulation uses only the full size mesh (no multigrid). 
After 100 iterations it has only reduced the maximum residual error to 0.87

    0 0.998475284655
    ...
    49 0.929067444026
    ...
    99 0.867532420282

The second test solves the same problem, but now using 5 levels of mesh resolution
(including the original). At each level 10 iterations of the Jacobi smoother
are performed (5 on the way down, 5 on the way up), and two V-cycles are performed.
In total then the same number of terations are performed as in the first test. 
The output is:

    Cycle  0 :  0.0338261789164
    Cycle  1 :  0.0022779802307

So after a single V-cycle the residual is reduced to 0.034, and to 0.0023 after two cycles.
This comparison based on iteration count is not really fair, because there is some overhead for the interpolations. 
On the other hand, most of the jacobi iterations are on a course mesh which is quicker than
an iteration on the full size mesh. 



