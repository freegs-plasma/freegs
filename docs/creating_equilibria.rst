Creating equilibria
===================

To generate a Grad-shafranov solution from scratch FreeGS needs
some physical parameters:

#. The locations of the coils
#. Plasma profiles (typically pressure and a current function) used to calculate toroidal current density :math:`J_\phi`
#. The desired locations of X-points, and constraints on the shape of the plasma.

and some numerical parameters:    
   
#. The domain in which we want to calculate the solution
#. The methods to be used to solve the equations


Tokamak coils, circuits and solenoid
------------------------------------
   
Example 1 (01-freeboundary.py) creates a simple lower single null
plasma from scratch. First import the freegs library

::
   
   import freegs

then create a tokamak, which specifies the location of the coils. In this example this is done using

::
   
   tokamak = freegs.machine.TestTokamak()

which creates an example machine with four poloidal field coils (two for the vertical field, and two for the radial field).
To define a custom machine, first create a list of coils:

::
   
   from freegs.machine import Coil
   
   coils = [("P1L", Coil(1.0, -1.1)),
             ("P1U", Coil(1.0, 1.1)),
             ("P2L", Coil(1.75, -0.6)),
             ("P2U", Coil(1.75, 0.6))]
   
Each tuple in the list defines the name of the coil (e.g. ``"P1L"``), then the corresponding object (e.g. ``Coil(1.0, -1.1)`` ).
Here ``Coil(R, Z)`` specifies the R (major radius) and Z (height) location of the coil in meters.

Create a tokamak by passing the list of coils to ``Machine``:

::
   
   tokamak = freegs.machine.Machine(coils)


Coil current control
~~~~~~~~~~~~~~~~~~~~

By default all coils can be controlled by the feedback system, but it may be that you want
to fix the current in some of the coils. This can be done by turning off control and setting the current:

::

   Coil(1.0, -1.1, control=False, current=50000.)

where the current is in Amps, and is for a coil with a single turn. Setting ``control=False``
removes the coil from feedback control.

Coil circuits
~~~~~~~~~~~~~

Usually not all coils in a tokamak are independently powered, but several coils
may be connected to the same power supply. This is handled in FreeGS using ``Circuit`` objects,
which consist of several coils. For example:

::

   from freegs.machine import Circuit
   
   Circuit( [("P2U", Coil(0.49,  1.76), 1.0),
             ("P2L", Coil(0.49, -1.76), 1.0)] )

This creates a ``Circuit`` by passing a list of tuples. Each tuple defines the coil name,
the ``Coil`` object (with R,Z location), and a current multiplier. In this case the current
multiplier is ``1.0`` for both coils, so the same current will flow in both coils. Alternatively
coils may be wired in opposite directions:

::

   Circuit( [("P6U", Coil(1.5,  0.9), 1.0),
             ("P6L", Coil(1.5, -0.9), -1.0)] )
   
so the current in coil "P6L" is in the opposite direction, but same magnitude, as the current in coil
"P6U".              

As with coils, circuits by default are controlled by the feedback system, and can be fixed by
setting ``control=False`` and specifying a current.

Solenoid
~~~~~~~~

Tokamaks typically operate with Ohmic current drive using a central solenoid. Flux leakage from
this solenoid can modify the equilibrum, particularly the locations of the strike points.
Solenoids are represented in FreeGS by a set of poiloidal coils:

::
   
   from freegs.machine import Solenoid
   
   solenoid = Solenoid(0.15, -1.4, 1.4, 100)

which defines the radius of the solenoid in meters (0.15m here), the lower and upper limits in Z (vertical position,
here :math:`\pm 1.4` m), and the number of poloidal coils to be used. These poloidal coils will be equally spaced between
the lower and upper Z limits.

As with ``Coil`` and ``Circuit``, solenoids can be removed from feedback control
by setting ``control=False`` and specifying a fixed current.

Mega-Amp Spherical Tokamak
~~~~~~~~~~~~~~~~~~~~~~~~~~

As an example, the definition of the Mega-Amp Spherical Tokamak (MAST) coilset is
given in the ``freegs.machine.MAST_sym()`` function:

::
   
   coils = [("P2", Circuit( [("P2U", Coil(0.49,  1.76), 1.0),
                              ("P2L", Coil(0.49, -1.76),1.0)] ))
            ,("P3", Circuit( [("P3U", Coil(1.1,  1.1), 1.0),
                              ("P3L", Coil(1.1, -1.1), 1.0)] ))
            ,("P4", Circuit( [("P4U", Coil(1.51,  1.095), 1.0),
                              ("P4L", Coil(1.51, -1.095), 1.0)] ))
            ,("P5", Circuit( [("P5U", Coil(1.66,  0.52), 1.0),
                              ("P5L", Coil(1.66, -0.52), 1.0)] ))
            ,("P6", Circuit( [("P6U", Coil(1.5,  0.9), 1.0),
                               ("P6L", Coil(1.5, -0.9), -1.0)] ))
            ,("P1", Solenoid(0.15, -1.45, 1.45, 100))
           ]

    tokamak = freegs.machine.Machine(coils)

This uses circuits "P2" to "P5" connecting pairs of upper and lower coils in series.
Circuit "P6" has its coils connected in opposite directions, so is used for vertical
position control. Finally "P1" is the central solenoid. Here all circuits and solenoid
are under position feedback control.

Equilibrium and plasma domain
-----------------------------

Having created a tokamak, an ``Equilibrium`` object can be created. This represents the
plasma solution.

::
   
   eq = freegs.Equilibrium(tokamak=tokamak,
                        Rmin=0.1, Rmax=2.0,    # Radial domain
                        Zmin=-1.0, Zmax=1.0,   # Height range
                        nx=65, ny=65)          # Number of grid points

In addition to the tokamak ``Machine`` object, this must be given the range of major radius
R and height Z (in meters), along with the radial (x) and vertical (y) resolution.
This resolution must be greater than 3, and is typically a power of 2 + 1 (:math:`2^n+1`), but
does not need to be. 

                        
Plasma profiles
---------------

