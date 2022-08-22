from freegs import machine, equilibrium, reconstruction, recon_tools, plotting
import numpy as np
show = True
check_limited = True
np.set_printoptions(threshold=np.inf)

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
x_r1 = 1.1
x_r2 = 1.1

tokamak = machine.EfitTestMachine(createVessel=True)

# Performing Reconstruction
print('Starting Reconstruction')
Recon = reconstruction.Reconstruction(tokamak, pprime_order, ffprime_order)

Recon.generate_Measurements(alpha_m, alpha_n)
Recon.take_Measurements_from_tokamak()
Recon.generate_Greens()
Recon.solve()

