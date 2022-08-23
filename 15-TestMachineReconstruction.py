from freegs import machine, equilibrium, reconstruction, plotting, boundary, jtor, control, picard
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
x_z2 = 0.6
x_r1 = 1.1
x_r2 = 1.1

tokamak = machine.EfitTestMachine()

eq = reconstruction.generate_Measurements(tokamak, alpha_m, alpha_n, x_z1=x_z1, x_z2=x_z2)

measurement_dict = {}
sigma_dict = {}
for sensor in tokamak.sensors:
    measurement_dict[sensor.name]=sensor.measurement
    sigma_dict[sensor.name]=sensor.measurement/sensor.weight

for name,coil in tokamak.coils:
    measurement_dict[name]=coil.current
    sigma_dict[name]=1e-5

# Performing Reconstruction
print('Starting Reconstruction')
Recon = reconstruction.Reconstruction(tokamak, pprime_order, ffprime_order)
Recon.solve_from_dictionary(measurement_dict, sigma_dict)

