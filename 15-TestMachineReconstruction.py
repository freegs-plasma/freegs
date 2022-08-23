from freegs import machine, equilibrium, reconstruction, plotting, boundary, jtor, control, picard
import numpy as np
show = True
check_limited = True
np.set_printoptions(threshold=np.inf)


# Creating an equilibrium
def generate_Measurements(tokamak, alpha_m, alpha_n, Rmin=0.1, Rmax=2, Zmin=-1,
                          Zmax=1, nx=65, ny=65, x_z1=0.6, x_z2=-0.6, x_r1=1.1,
                          x_r2=1.1):
    """
    Function for running the forward simulation

    Parameters
    ----------
    tokamak - tokamak machine
    alpha_m - coefficients to model the forward equilibrium
    alpha_n - more info found in FreeGS documentation
    Rmin - Minimum position for R grid
    Rmax - Maximum position for R grid
    Zmin - Minimum position for Z grid
    Zmax - Maximum position for Z grid
    nx - number of points in radial direction
    ny - number of points in vertical direction
    x_z1 - z position of upper x point
    x_z2 - z position of lower x point
    x_r1 - r position of upper x point
    x_r2 - r position of lower x point
    show - Option for plotting the function

    Returns
    -------
    tokamak.sensors - returns machine sensors with measurements contained within
    tokamak.coils - returns machine coils with currents contained within
    eq - equilibrium object

    """

    eq = equilibrium.Equilibrium(tokamak=tokamak,
                                 Rmin=Rmin, Rmax=Rmax,  # Radial domain
                                 Zmin=Zmin, Zmax=Zmax,  # Height range
                                 nx=nx, ny=ny,  # Number of grid points
                                 boundary=boundary.freeBoundary)  # Boundary condition

    profiles = jtor.ConstrainPaxisIp(eq, 1e3,
                                     2e5,
                                     2.0,
                                     alpha_m=alpha_m,
                                     alpha_n=alpha_n
                                     )

    xpoints = [(x_r1, x_z2),  # (R,Z) locations of X-points
               (x_r2, x_z1)]
    isoflux = [(x_r1, x_z2, x_r2, x_z1)]  # (R1,Z1, R2,Z2) pair of locations

    constrain = control.constrain(xpoints=xpoints, isoflux=isoflux)

    picard.solve(eq,  # The equilibrium to adjust
                 profiles,  # The toroidal current profile function
                 constrain,
                 show=show,
                 check_limited=True)

    print('Construction Diagnostics')
    tokamak.printMeasurements(equilibrium=eq)
    tokamak.printCurrents()
    return eq

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

eq = generate_Measurements(tokamak, alpha_m, alpha_n)

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
Recon.solve_from_tokamak()

