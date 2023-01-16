"""
This script simulates an equilibrium using the forward solver in freegs,
then uses the sensor measurments and coil currents to reconstruct the data
"""
from freegs import machine, reconstruction, plotting

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

eq_setup = {'tokamak': tokamak, 'Rmin': Rmin, 'Rmax': Rmax, 'Zmin': Zmin, 'Zmax': Zmax, 'nx': nx, 'ny': ny}
Recon = reconstruction.Reconstruction(pprime_order, ffprime_order, tolerance=1e-10,  **eq_setup)
Recon.solve_from_tokamak()
plotting.plotEquilibrium(Recon)