Multigrid example
=================

Solves a Laplacian in a square domain, with Dirichlet (zero value)
boundary conditions. The smoother is a simple Jacobi method, and 2nd order
central differences is used to discretise the operator. 

Usage
-----

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



