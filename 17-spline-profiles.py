#!/usr/bin/env python

import freegs
import numpy as np
import matplotlib.pyplot as plt

#########################################
# Create the machine, which specifies coil locations
# and equilibrium, specifying the domain to solve over

tokamak = freegs.machine.TestTokamak()

eq = freegs.Equilibrium(tokamak=tokamak,
                        Rmin=0.1, Rmax=2.0,    # Radial domain
                        Zmin=-1.0, Zmax=1.0,   # Height range
                        nx=65, ny=65,          # Number of grid points
                        boundary=freegs.boundary.freeBoundaryHagenow)  # Boundary condition


#########################################
# Plasma profiles

profiles = freegs.jtor.ConstrainBetapIp(eq,
                                        3.214806e-02, # Poloidal beta
                                        2e5, # Plasma current [Amps]
                                        2.0) # Vacuum f=R*Bt

#########################################
# Coil current constraints
#
# Specify locations of the X-points
# to use to constrain coil currents

xpoints = [(1.1, -0.6),   # (R,Z) locations of X-points
           (1.1, 0.8)]

isoflux = [(1.1,-0.6, 1.1,0.6)] # (R1,Z1, R2,Z2) pair of locations

constrain = freegs.control.constrain(xpoints=xpoints, isoflux=isoflux)

#########################################
# Nonlinear solve

freegs.solve(eq,          # The equilibrium to adjust
             profiles,    # The toroidal current profile function
             constrain,
             show=True)   # Constraint function to set coil currents

# eq now contains the solution

print("Done!")

print("Plasma current: %e Amps" % (eq.plasmaCurrent()))
print("Plasma pressure on axis: %e Pascals" % (eq.pressure(0.0)))
print("Poloidal beta: %e" % (eq.poloidalBeta()))

# Now, extract the pprime and ffprime profiles from the solution
psi_n_data = np.linspace(0.0,1.0,101,endpoint=True)
pprime_data = eq.pprime(psi_n_data)
ffprime_data = eq.ffprime(psi_n_data)

# Next, solve again, this time using the above data as a placeholder for bespoke profile data.
# In user-specific implementations this data may be obtained elsewhere, e.g. from MDSplus or
# from the output of another code.

print('Using splined profiles for pprime and ffprime')

#########################################
# Create the machine, which specifies coil locations
# and equilibrium, specifying the domain to solve over

tokamak2 = freegs.machine.TestTokamak()

eq2 = freegs.Equilibrium(tokamak=tokamak2,
                        Rmin=0.1, Rmax=2.0,    # Radial domain
                        Zmin=-1.0, Zmax=1.0,   # Height range
                        nx=65, ny=65,          # Number of grid points
                        boundary=freegs.boundary.freeBoundaryHagenow)  # Boundary condition


#########################################
# Plasma profiles

profiles2 = freegs.jtor.BetapIpConstrainedSplineProfiles(eq,
                                        3.214806e-02, # Poloidal beta
                                        2e5, # Plasma current [Amps]
                                        1.0, # Raxis [m],
                                        psi_n_data, # Spline data for normalised psi
                                        pprime_data, # Spline data for pprime
                                        ffprime_data, # Spline data for ffprime
                                        2.0) # Vacuum f=R*Bt

#########################################
# Coil current constraints
#
# Specify locations of the X-points
# to use to constrain coil currents

xpoints = [(1.1, -0.6),   # (R,Z) locations of X-points
           (1.1, 0.8)]

isoflux = [(1.1,-0.6, 1.1,0.6)] # (R1,Z1, R2,Z2) pair of locations

constrain2 = freegs.control.constrain(xpoints=xpoints, isoflux=isoflux)

#########################################
# Nonlinear solve

freegs.solve(eq2,          # The equilibrium to adjust
             profiles2,    # The toroidal current profile function
             constrain2,
             show=True)   # Constraint function to set coil currents

# eq2 now contains the solution

print("Done!")

print("Plasma current: %e Amps" % (eq2.plasmaCurrent()))
print("Plasma pressure on axis: %e Pascals" % (eq2.pressure(0.0)))
print("Poloidal beta: %e" % (eq2.poloidalBeta()))

# Compare results - parameterised vs splined profiles
fig, ax = plt.subplots(1,3)

ax[0].contour(eq.R,eq.Z,eq.psi(),levels=[eq.psi_bndry],colors='r')
ax[0].contour(eq2.R,eq2.Z,eq2.psi(),levels=[eq2.psi_bndry],colors='b')
ax[0].set_aspect('equal')
ax[0].set_xlabel('R (m)')
ax[0].set_ylabel('Z (m)')

ax[1].plot(psi_n_data,eq.pprime(psi_n_data),color='r')
ax[1].plot(psi_n_data,eq2.pprime(psi_n_data),color='b')
ax[1].set_xlabel(r'$\psi_{N}$')
ax[1].set_ylabel(r'pprime($\psi_{N}$)')

ax[2].plot(psi_n_data,eq.ffprime(psi_n_data),color='r')
ax[2].plot(psi_n_data,eq2.ffprime(psi_n_data),color='b')
ax[2].set_xlabel(r'$\psi_{N}$')
ax[2].set_ylabel(r'ffprime($\psi_{N}$)')

plt.show()