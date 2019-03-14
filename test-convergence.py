#!/usr/bin/env python

import freegs
import numpy as np
import matplotlib.pyplot as plt

start_resolution = 17
nrefinements = 2 # Number of refinements 
rtol = 1e-10     # Relative tolerance in Picard iteration
location = (1.2, 0.1)  # Location to record values at

############################################
# Generate low resolution solution

tokamak = freegs.machine.TestTokamak()

eq = freegs.Equilibrium(tokamak=tokamak,
                        Rmin=0.1, Rmax=2.0,    # Radial domain
                        Zmin=-1.0, Zmax=1.0,   # Height range
                        nx=start_resolution, ny=start_resolution, # Number of grid points
                        boundary=freegs.boundary.freeBoundaryHagenow) 

profiles = freegs.jtor.ConstrainPaxisIp(1e3, # Plasma pressure on axis [Pascals]
                                        2e5, # Plasma current [Amps]
                                        2.0) # Vacuum f=R*Bt

xpoints = [(1.1, -0.6),   # (R,Z) locations of X-points
           (1.1, 0.8)]

isoflux = [(1.1,-0.6, 1.1,0.6)] # (R1,Z1, R2,Z2) pair of locations

constrain = freegs.control.constrain(xpoints=xpoints, isoflux=isoflux)

freegs.solve(eq,          # The equilibrium to adjust
             profiles,    # The toroidal current profile function
             constrain,
             rtol=rtol)

############################################
# Now have initial solution

resolutions = [eq.R.shape[0]]
psivals = [eq.psiRZ(*location)]
brvals = [eq.Br(*location)]
volumevals = [eq.plasmaVolume()]
coilcurrents = [eq.tokamak["P1L"].current]

for i in range(nrefinements):

    print("\n========== Refining ===========\n")
    
    # Increase resolution
    eq = freegs.equilibrium.refine(eq)


    
    # Re-solve
    freegs.solve(eq,
                 profiles,    # The toroidal current profile function
                 constrain,
                 rtol=rtol)
    
    resolutions.append(eq.R.shape[0])
    psivals.append(eq.psiRZ(*location))
    brvals.append(eq.Br(*location))
    volumevals.append(eq.plasmaVolume())
    coilcurrents.append(eq.tokamak["P1L"].current)


resolutions = np.squeeze(np.array(resolutions))
psivals = np.squeeze(np.array(psivals))
brvals = np.squeeze(np.array(brvals))
volumevals = np.squeeze(np.array(volumevals))
coilcurrents = np.squeeze(np.array(coilcurrents))

fig, axes = plt.subplots(2, 2, sharex=True)

def plot_convergence(axis, values, title):
    # Absolute differences
    diffs = np.abs(values[1:] - values[0:-1])

    # Convergence order. Note one shorter than diffs array
    orders = np.log(diffs[:-1]/diffs[1:]) / np.log(2.)

    axis.plot(resolutions[1:], diffs)
    axis.set_title(title)
    axis.set_yscale("log")
    axis.set_xscale("log")

    for x, y, order in zip(resolutions[2:], diffs[1:], orders):
        axis.text(x, y, "{:.1f}".format(order))

plot_convergence(axes[0,0], psivals, "Flux at {}".format(location)) 
plot_convergence(axes[1,0], brvals, "Br at {}".format(location))
plot_convergence(axes[0,1], volumevals, "Plasma volume")
plot_convergence(axes[1,1], coilcurrents, "P1L coil current")

plt.savefig("test-convergence.pdf")
plt.savefig("test-convergence.png")

plt.show()

