Free boundary Grad-Shafranov solver
===================================

This Python module solves the Grad-Shafranov equation for axisymmetric
plasma equilibria, mainly for free boundary calculations.

    Copyright 2016 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.


Examples
--------

The Jupyter notebooks contain examples wuth additional notes

* MAST-example.ipynb 

There are also some Python scripts to run short tests
and examples

    $ python 01-freeboundary.py

This solves a free boundary problem, specifying the desired location of two X-points.
Writes the equilibrium to a G-EQDSK file "lsn.geqdsk"

    $ python 02-read-geqdsk.py

Reads in the file "lsn.geqdsk", inferring the coil currents from the plasma boundary
and profiles in the G-EQDSK file.

    $ python 03-mast.py

Calculates a double-null (CDND) equilibrium for MAST from scratch. Writes solution to
G-EQDSK file "mast.geqdsk"

    $ python 04-read-mast-geqdsk.py

Reads the file "mast.geqdsk", inferring the coil currents.

    $ python 05-fixed-boundary.py 

This example solves a fixed boundary problem, in which the square edges of the domain
are fixed. The plasma pressure on axis and plasma current are fixed.

    $ python 06-xpoints.py

This demonstrates the coil current control code, finding X-points, and marking core region
These routines are used inside the free boundary solver

Files
-----

The "freegs" module consists of the following files:

boundary.py        - Operators for applying boundary conditions to plasma psi
control.py         - Routines for controlling coil currents based on constraints
critical.py        - Finds critical points (O- and X-points)
equilibrium.py     - Represents the plasma equilibrium state
gradshafranov.py   - Greens functions and operators for the Grad-Shafranov equation
jtor.py            - Routines for calculating toroidal current density (profiles)
machine.py         - Represents the coils and power supply circuits
multigrid.py       - The multigrid solver for the linear elliptic operator
picard.py          - Nonlinear solver, iterating the profiles and constraints
plotting.py        - Plotting routines using matplotlib

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



