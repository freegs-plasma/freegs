Input and Output
================

A standard format for storing tokamak equilibrium data is `G-EQDSK <https://fusion.gat.com/theory/Efitgeqdsk>`_
which contains the poloidal flux in (R,Z) and 1D profiles of pressure, :math:`f=RB_\phi`, safety factor q,
and other quantities related to the Grad-Shafranov solution. The G-EQDSK format does not however have a standard
for specifying the location of, and currents in, the poloidal field coils. This makes writing G-EQDSK files quite
straightforward, but reading them more challenging, as these coil currents must be inferred.

The implementation of the file input and output is divided into a high level interface in ``freegs.geqdsk`` and a low level interface in ``freegs._geqdsk``. The high level interface handles ``Equilibrium`` objects, whilst the low level interface handles
simple dictionaries.

Writing G-EQDSK files
---------------------

Import the ``geqdsk`` module from ``freegs``, passing an
``Equilibrium`` object and a file handle:

::

   from freegs import geqdsk
   
   with open("lsn.geqdsk", "w") as f:
     geqdsk.write(eq, f)



Reading G-EQDSK files
---------------------

This is complicated by the need to infer the currents in the coils. To do this the locations of the coils need to be specified. An example is ``02-read-geqdsk.py`` which reads a file produced by ``01-freeboundary.py``. First create a machine object which specifies the location of the coils

::

   from freegs import machine
   tokamak = machine.TestTokamak()

Reading the file then consists of
   
::

   from freegs import geqdsk
   with open("lsn.geqdsk") as f:
     eq = geqdsk.read(f, tokamak, show=True)


This ``read`` function has the following stages:
     
#. Reads the plasma state from the file into an Equilibrium object
#. Uses the control system to find starting values for the coil currents, keeping the plasma boundary and X-point locations fixed
#. Runs the Grad-Shafranov picard solver, keeping profiles and boundary shape fixed. This adjusts the plasma solution and coil currents to find a self-consistent solution. 

The ``show`` optional parameter displays a plot of the equilibrium, and shows the stages in the Grad-shafranov solve.

Some options are:

#. `domain = (Rmin, Rmax, Zmin, Zmax)` which can be used to specify
   the (R,Z) domain to solve on. This is useful for reading inputs
   from fixed boundary codes like SCENE, where the X-point(s) may lie
   outside the domain. The coil current feedback needs to find where
   the edge of the plasma is, so needs to find an X-point to work
   correctly. The grid spacing is never allowed to increase, so this
   may result in an increase in the number of grid points.
#. `blend` is a number between 0 and 1, and is used in the nonlinear
   Picard iteration. It determines what fraction of the previous
   solution is used in the next solution. The default is 0, so no
   blending is done. Adding blending (e.g. 0.5, 0.7) usually slows
   convergence, but can stabilise oscillating or unstable solutions.
#. `fit_sol` is `False` by default, so only points inside the
   separatrix are used when fitting the coil currents to match the
   input poloidal flux. This is useful in cases like reading SCENE
   inputs, where the poloidal flux in the Scrape-Off Layer (SOL) is
   not valid in the input. Setting `fit_sol=True` causes the whole
   input domain to be used in the fitting. Currently this is NOT
   compatible with setting a domain. It is useful when the locations
   of strike points need to be matched, and may better constrain coil
   currents for free boundary inputs (e.g. EFIT).

Specifying coil currents
~~~~~~~~~~~~~~~~~~~~~~~~

A feedback control system is used to keep the plasma boundary and X-point locations fixed whilst adjusting the coil currents.
If additional information about coil currents is available, then this can be used to fix some or all of the coil currents.

To see a list of the coils available:

::

   print(tokamak.coils)

   [('P1L', Coil(R=1.0,Z=-1.1,current=0.0,control=True)),
   ('P1U', Coil(R=1.0,Z=1.1,current=0.0,control=True)),
   ('P2L', Coil(R=1.75,Z=-0.6,current=0.0,control=True)),
   ('P2U', Coil(R=1.75,Z=0.6,current=0.0,control=True))]

   
Before calling ``geqdsk.read``, specify the coil currents in the ``tokamak`` object:

::

   tokamak["P1L"].current = 5e4  # Amp-turns

This will give the control system a starting value for the coil currents, but since the coil is still under feedback control it may still be altered. To fix the current in the coil turn off control:

::

   tokamak["P1L"].control = False  # No feedback control (fixed current)


Writing DivGeo files
--------------------

These are used as input to the SOLPS grid generator. They contain a subset
of what's in G-EQDSK files, with no plasma pressure or current
profiles. 

To convert a G-EQDSK file directly to DivGeo without any solves, and
without needing to know the coil locations, use the low-level
routines::

  from freegs import _geqdsk
  from freegs import _divgeo

  with open("input.geqdsk") as f:
    data = _geqdsk.read(f)
  
  with open("output.equ", "w") as f:
    _divgeo.write(data, f)

FreeGS equilibria objects can also be written to DivGeo files using
the `divgeo` module::

  from freegs import geqdsk
  from freegs import divgeo
  from freegs.machine import TestTokamak

  # Read a G-EQDSK file
  with open("lsn.geqdsk") as f:
    eq = geqdsk.read(f, TestTokamak(), show=True)

  # Modify the equilbrium...

  # Save to DivGeo
  with open("lsn.equ", "w") as f:
    divgeo.write(eq, f)

