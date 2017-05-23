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
here :math:`\pm` 1.4 m), and the number of poloidal coils to be used. These poloidal coils will be equally spaced between
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
plasma solution, and contains the tokamak with the coil currents.

::
   
   eq = freegs.Equilibrium(tokamak=tokamak,
                        Rmin=0.1, Rmax=2.0,    # Radial domain
                        Zmin=-1.0, Zmax=1.0,   # Height range
                        nx=65, ny=65)          # Number of grid points

In addition to the tokamak ``Machine`` object, this must be given the range of major radius
R and height Z (in meters), along with the radial (x) and vertical (y) resolution.
This resolution must be greater than 3, and is typically a power of 2 + 1 (:math:`2^n+1`) for efficiency, but
does not need to be. 

                        
Plasma profiles
---------------

The plasma profiles, such as pressure or safety factor, are used to determine the toroidal current :math:`J_\phi`:

.. math::

   J_\phi\left(R,Z\right) = R\frac{\partial p\left(\psi\right)}{\partial \psi} + \frac{f\left(\psi\right)}{R\mu_0}\frac{\partial f\left(\psi\right)}{\partial \psi}

where the flux function :math:`p\left(\psi\right)` is the plasma pressure (in Pascals), and :math:`f\left(\psi\right) = RB_\phi` is the poloidal current function.
Classes and functions to handle these profiles are in ``freegs.jtor``


Constrain pressure and current
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of the most intuitive methods is to fix the shape
of the plasma profiles, and adjust them to fix the
pressure on the magnetic axis and total plasma current.
To do this, create a ``ConstrainPaxisIp`` profile object:

::
   
   profiles = freegs.jtor.ConstrainPaxisIp(1e4, # Pressure on axis [Pa]
                                           1e6, # Plasma current [Amps]
                                           1.0) # Vacuum f=R*Bt


This sets the toroidal current to:

.. math::

   J_\phi = L \left[\beta_0 R + \left(1-\beta_0\right)/R\right] \left(1-\psi_n^{\alpha_m}\right)^{\alpha_n}

where :math:`\psi_n` is the normalised poloidal flux, 0 on the magnetic axis and 1 on the plasma boundary/separatrix.
The constants which determine the profile shapes are :math:`\alpha_m = 1` and  :math:`\alpha_n = 2`. These can be changed by specifying in the initialisation of ``ConstrainPaxisIp``.

The values of :math:`L` and :math:`\beta_0` are determined from the constraints: The pressure on axis is given by integrating the pressure gradient flux function 

.. math::
   
   p_{axis} = - L \beta_0 R \int_{axis}^{boundary}\left(1-\psi_n^{\alpha_m}\right)^{\alpha_n} d\psi

The total toroidal plasma current is calculated by integrating the toroidal current function over the 2D domain:

.. math::
   
   I_p = L\beta_0 \iint R \left(1-\psi_n^{\alpha_m}\right)^{\alpha_n} dR dZ + L\left(1-\beta_0\right)\iint \frac{1}{R} \left(1-\psi_n^{\alpha_m}\right)^{\alpha_n} dR dZ
   

The integrals in these two constraints are done numerically,
and then rearranged to get :math:`L` and :math:`\beta_0`. 
   
Constrain poloidal beta and current
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is a variation which replaces the constraint on pressure with a constraint on poloidal beta:

.. math::

   \beta_p = \frac{8\pi}{\mu_0} \frac{1}{I_p^2}\iint p\left(\psi\right) dRdZ 

This is the method used in `Y.M.Jeon 2015 <https://arxiv.org/abs/1503.03135>`_, on which the profile choices here are based.

::
   
   profiles = freegs.jtor.ConstrainBetapIp(0.5, # Poloidal beta
                                           1e6, # Plasma current [Amps]
                                           1.0) # Vacuum f=R*Bt
   
By integrating over the plasma domain and combining the constraints on poloidal beta and plasma current, the values of :math:`L` and :math:`\beta_0` are found.

Feedback and shape control
--------------------------

To determine the currents in the coils, the shape and position of the plasma needs to be constrained. In addition, diverted tokamak plasmas are inherently vertically unstable, and need vertical position feedback to maintain a stationary equilibrium.
If vertical position is not constrained, then free boundary equilibrium solvers can also become vertically unstable. A typical symptom is that each nonlinear iteration of the solver results in a slightly shifted or smaller plasma, until the plasma hits the boundary, disappears, or forms unphysical shapes causing the solver to fail.

Currently the following kinds of constraints are implemented:

* X-point constraints adjust the coil currents so that X-points (nulls in the poloidal field) are formed at the locations requested. 

* Isoflux constraints adjust the coil currents so that the two locations specified have the same poloidal flux. This usually means they are on the same flux surface, but not necessarily.

* Psi value constraints, which adjust the coil currents so that given locations have the specified flux. 
    
As an example, the following code creates a feedback control with two X-point constraints and one isoflux constraint:

::

   xpoints = [(1.1, -0.6),   # (R,Z) locations of X-points
              (1.1, 0.8)]

   isoflux = [(1.1,-0.6, 1.1,0.6)] # (R1,Z1, R2,Z2) pairs 

   constrain = freegs.control.constrain(xpoints=xpoints, isoflux=isoflux)

The control system determines the currents in the coils which are under feedback control, using the given constraints.
There may be more unknown coil currents than constraints, or more constraints than coil currents. There may therefore be either no solution or many solutions to the constraint problem.
Here Tikhonov regularisation is used to produce a unique solution and penalise large coil currents. 

Solving
-------

To solve the Grad-Shafranov equation to find the free boundary solution, call ``freegs.solve``:

::
   
   freegs.solve(eq,          # The equilibrium to adjust
                profiles,    # The toroidal current profile 
                constrain)   # Feedback control


This call modifies the input equilibrium (eq), finding a solution
based on the given plasma profiles and shape control.

The Grad-Shafranov equation is nonlinear, and is solved using Picard iteration. 
