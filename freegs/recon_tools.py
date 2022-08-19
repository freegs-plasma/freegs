import numpy as np
from . import critical, plotting, machine, jtor, control, picard, boundary, equilibrium, filament_coil
import scipy
from shapely.geometry import Point, Polygon, LineString, LinearRing

# Running forward simulation
def get_values(tokamak,alpha_m, alpha_n, Rmin=0.1, Rmax=2, Zmin=-1, Zmax=1,nx=65,ny=65,x_z1=0.6,x_z2=-0.6,x_r1=1.1,x_r2=1.1, show=True, check_limited=True):
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
    profiles.Beta0 - returns simulated beta0
    profiles.L - returns simulated L
    profiles.mask - returns final mask
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
                 check_limited=check_limited)

    print('Construction Diagnostics')
    if tokamak.sensors != None:
        tokamak.printMeasurements(equilibrium=eq)

    tokamak.printCurrents()


    return tokamak.sensors, tokamak.coils, eq

def give_values(tokamak, sensors,coils):
    """
    Function called to generate measurement matrix and sigma
    Also give new tokamak object the coil currents

    Parameters
    ----------
    tokamak - tokamak object
    sensors - tokamak sensors from forward simulation
    coils - tokamak coils from forward simulation

    Returns
    -------
    M - Returns measurement matrix
    sigma - Returns uncertainty values
    """
    M = []
    sigma = []

    for sensor in sensors:
        M.append([sensor.measurement])
        if isinstance(sensor, machine.RogowskiSensor):
            if sensor.measurement==0:
                sigma.append([1])
            else:
                sigma.append([sensor.measurement/ sensor.weight])
        else:
            sigma.append([sensor.measurement/ sensor.weight])

    # Adding Coil Currents To Machine and M
    for i in range(len(tokamak.coils)):
        coil = coils[i][1]
        M.append([coil.current])
        sigma.append([0.00001])

    return M, sigma

# Grid Conversion Functions
def grid_to_line(Grid):
    """
    Takes a grid creates a line

    Parameters
    ----------
    Grid - input Grid to be converted

    Returns
    -------
    Line - returns the grid in line form
    """
    nx = Grid.shape[0]
    ny = Grid.shape[1]
    Line = np.zeros(nx * ny)
    for j in range(ny):
        for i in range(nx):
            Line[j * nx + i] = Grid[i, j]
    return Line

def line_to_grid(Line, nx, ny):
    """
    Takes a line and creates a grid

    Parameters
    ----------
    Line - input the line to be converted


    Returns
    -------
    Grid - returns Grid from line
    """
    assert len(Line) == nx * ny
    Grid = np.zeros((nx, ny))
    for j in range(ny):
        for i in range(nx):
            #breakpoint()
            Grid[i, j] = Line[j * nx + i]
    return Grid


# Perform chi squared test for convergence
def chi_squared_test(M, sigma, H):
    """
    Performs chi squared test on measurements and computed measurements

    Parameters
    ----------
    M - measurements
    sigma - measurement uncertainties
    H - computed measurements

    Returns
    -------
    cs - chi squared value
    """
    cs = 0
    for i in range(len(M)):
        cs += ((M[i] - H[i]) / sigma[i])**2
    return cs


# Calculating the plasma contirbution to the measurements
def get_M_plasma(M, tokamak):
    """
    Finds the plasma contribution to the measurements

    Parameters
    ----------
    M - measurements
    tokamak - tokamak

    Returns
    -------
    M_plasma
    """
    M_plasma = []
    tokamak.takeMeasurements()
    for i in range(len(M)):
        sensor = tokamak.sensors[i]
        M_plasma.append([M[i][0] - sensor.measurement])
    return M_plasma


# Calculate the Current initialisation T matrix
def current_initialise(M, tokamak, eq, T=None):
    """
    Function for calculating the initial jtor

    Parameters
    ----------
    G - greens matrix
    M_plasma - plasma contribution to measurements
    eq - equilibrium object
    T - Initialisation basis matrix

    Returns
    -------
    jtor - current density matrix
    """

    M_plasma = get_M_plasma(M,tokamak)

    if not isinstance(T, np.ndarray):
        T = get_T(eq,5,5)

    coefs = scipy.linalg.lstsq(tokamak.Gplasma@T, M_plasma)[0]
    jtor = T@coefs
    return jtor


# Blend Psi Matrices
def blender(x_new,x_old, blend=0.4):
    """
    Function to blend the old psi with the new psi

    Parameters
    ----------
    x_new - new normalised psi
    x_old - old normalised psi
    blend - blending coefficient

    Returns
    -------
    x - blended psi
    """
    x = (1-blend)*x_new + blend*x_old
    return x



# Printing Functions
def print_frac_error(tokamak, H):
    print(' ')
    print('Fractional Error between H and Sensors')
    print('==========================')
    for i in range(len(tokamak.sensors)):
        sensor = tokamak.sensors[i]
        hval = H[i]
        print((hval - sensor.measurement) / hval)

def print_H(H):
    print(' ')
    print('Computed Values')
    print('==========================')
    print(H)
