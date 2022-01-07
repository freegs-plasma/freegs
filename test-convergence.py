#!/usr/bin/env python

import freegs
import numpy as np
import matplotlib.pyplot as plt

start_resolution = 17
nrefinements = 5 # Number of refinements. Minimum 2
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
             rtol=rtol,
             maxits = 120)

############################################
# Now have initial solution

resolutions = [eq.R.shape[0]]
psivals = [eq.psiRZ(*location)]
brvals = [eq.Br(*location)]
volumevals = [eq.plasmaVolume()]
coilcurrents = [eq.tokamak["P1L"].current]

# List of l2 and l∞ norms
l2vals = []
linfvals = []

for i in range(nrefinements):

    print("\n========== Refining ===========\n")

    psi_old = eq.psi()
    R_old = eq.R
    Z_old = eq.Z
    
    # Increase resolution
    eq = freegs.equilibrium.refine(eq)
    
    # Re-solve
    freegs.solve(eq,
                 profiles,
                 constrain,
                 rtol=rtol,
                 maxits=120)

    resolutions.append(eq.R.shape[0])
    
    # Get the new psi on the old points
    psi_new = eq.psiRZ(R_old, Z_old)

    # Global norms of the change in psi
    l2 = np.sqrt(np.mean( (psi_new - psi_old)**2 ))
    linf = np.amax(np.abs( psi_new - psi_old ))

    l2vals.append(l2)
    linfvals.append(linf)

    # Point-wise values
    psivals.append(eq.psiRZ(*location))
    brvals.append(eq.Br(*location))

    # Coil current
    coilcurrents.append(eq.tokamak["P1L"].current)

    # Global quantity
    volumevals.append(eq.plasmaVolume())


resolutions = np.squeeze(np.array(resolutions))
psivals = np.squeeze(np.array(psivals))
brvals = np.squeeze(np.array(brvals))
volumevals = np.squeeze(np.array(volumevals))
coilcurrents = np.squeeze(np.array(coilcurrents))

l2vals = np.array(l2vals)
linfvals = np.array(linfvals)

fig, axes = plt.subplots(2, 3, sharex=True)

def plot_convergence(axis, title, values=None, diffs=None):
    # Absolute differences
    if diffs is None:
        diffs = np.abs(values[1:] - values[0:-1])

    # Convergence order. Note one shorter than diffs array
    orders = np.log(diffs[:-1]/diffs[1:]) / np.log(2.)

    axis.plot(resolutions[1:]-1, diffs)
    axis.set_title(title)
    axis.set_yscale("log")
    axis.set_xscale("log")

    for x, y, order in zip(resolutions[2:], diffs[1:], orders):
        axis.text(x, y, "{:.1f}".format(order))

    resolution_list = list(resolutions[1:]-1)
    axis.set_xticks(resolution_list)
    axis.set_xticklabels(["{:d}".format(r) for r in resolution_list])
    axis.set_xticks([], minor=True)
    

plot_convergence(axes[0,0], r"Change in $\psi$ at {}".format(location), values=psivals)
plot_convergence(axes[0,1], r"$\psi$ difference $l_2$ norm", diffs=l2vals)
plot_convergence(axes[0,2], r"$\psi$ difference $l_\infty$ norm", diffs=linfvals)
plot_convergence(axes[1,0], "Br at {}".format(location), values=brvals)
plot_convergence(axes[1,1], "Plasma volume", values=volumevals)
plot_convergence(axes[1,2], "P1L coil current", values=coilcurrents)

plt.savefig("test-convergence.pdf")
plt.savefig("test-convergence.png")

plt.show()

