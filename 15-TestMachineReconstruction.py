from freegs import machine, reconstruction
import numpy as np

# Defining equilibrium grid
Rmin = 0.1
Rmax = 2
Zmin = -1
Zmax = 1
nx = 65
ny = 65

# Defining initial model and reconstruction parameters
alpha_m = 1
alpha_n = 2
pprime_order = 3
ffprime_order = 3
x_z1 = -0.6
x_z2 = 0.7
x_r1 = 1
x_r2 = 1

tokamak = machine.EfitTestMachine()

eq = reconstruction.generate_Measurements(tokamak, alpha_m, alpha_n, x_z1=x_z1, x_z2=x_z2, x_r1=x_r1, x_r2=x_r2)

# Performing Reconstruction
print('Starting Reconstruction')
Recon = reconstruction.Reconstruction(tokamak, pprime_order, ffprime_order, tolerance=1e-20, use_VesselCurrents=False)
Recon.solve_from_tokamak()