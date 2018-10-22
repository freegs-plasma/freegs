Diagnostics
===========

Once an equilibrium has been generated (see `Creating equilibria`_)
there are routines for diagnosing and calculating derived quantities.
Here the ``Equilibrium`` object is assumed to be called ``eq`` and the
``Tokamak`` object called ``tokamak``.

Safety factor, q
----------------

The safety factor at a given normalised poloidal flux (0 on the
magnetic axis, 1 at the separatrix) can be calculated using the
``q(psinorm)`` function::

  eq.q(0.9)  # safety factor at psi_norm = 0.9

Separatrix location
-------------------

A set of points on the separatrix, measured in meters::

  RZ = eq.separatrix()

  R = RZ[:,0]
  Z = RZ[:,1]

  import matplotlib.pyplot as plt
  plt.plot(R,Z)



Currents in the coils
---------------------

The currents in all coils can be printed using::

  tokamak.printCurrent()


Forces on the coils
-------------------

The forces on all poloidal field coils can be calculated and returned
as a dictionary::

  eq.getForces()

or formatted and printed::

  eq.printForces()

These forces on the poloidal coils are due to a combination of:

- The magnetic fields due to other coils
- The magnetic field of the plasma
- A self (hoop) force due to the coil's own current

The self force is the most difficult to calculate, since the force
depends on the cross-section of the coil. The formula used is from
`David A. Garren and James Chen (1998) <https://doi.org/10.1063/1.870491>`_.
For a circular current loop of radius :math:`R` and minor radius
:math:`a`, the outward force per unit length is:

.. math::
   f = \frac{\mu_0 * I^2}{4\pi R} \left(ln(8*R/a) - 1 + \xi_i/2\right)

where :math:`\xi_i` is a constant which depends on the internal
current distribution. For a constant, uniform current :math:`\xi_i = 1/2`;
for a rapidly varying surface current :math:`\xi_i = 0`.

For the purposes of calculating this force the cross-section is
assumed to be circular. The area can be set to a fixed value::

  tokamak["P1L"].area = 0.01  # Area in m^2

where here "P1L" is the label of the coil. The default is
to calculate the area using a limit on the maximum
current density. A typical value chosen here for Nb3Sn superconductor
is :math:`3.5\times 10^9 A/m^2`, taken from
`Kalsi (1986) <https://doi.org/10.1016/0167-899X(86)90010-8>`_ .


This can be changed e.g::

  from freegs import machine
  
  tokamak["P1L"].area = machine.AreaCurrentLimit(1e9)

would set the current limit for coil "P1L" to 1e9 Amps per square meter.
