import numpy as np
import math
from . import critical, plotting, machine
from .gradshafranov import Greens, GreensBr, GreensBz
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from .recon_tools import chi_squared_test
from scipy.special import ellipe, ellipk
mu0 = 4e-7 * np.pi

# Basis Matrix B (N x nc)
def get_B(x, eq, pprime_order, ffprime_order,c=None,VC=False):
    """
    Function for calculating the basis matrix, runs every iteration

    Parameters
    ----------
    x - normalised psi
    eq - equilibrium object
    R - matrix of radial values
    pprime_order - number of polynomial coefficients for pprime model
    ffprime_order - number of polynomial coefficients for ffprime model
    c - coefficients matrix
    VC - Vertical Control Option

    Returns
    -------
    B - Basis matrix
    """
    # Create Empty Matrix
    N = eq.nx * eq.ny
    R = eq.R.flatten(order='F')
    nc = pprime_order + ffprime_order
    dR = eq.R[1, 0] - eq.R[0, 0]
    dZ = eq.Z[0, 1] - eq.Z[0, 0]


    if VC:
        B = np.zeros((N, nc+1))
        for i in range(pprime_order):
            for j in range(N):

                B[j, i] = R[j] * (x[j])**i

        # ff' relations
        for i in range(ffprime_order):
            for j in range(N):
                B[j, i + pprime_order] = (1 / (mu0 * R[j])) * (x[j])**i

        x_z = np.gradient(np.reshape(x, (eq.nx, eq.ny), order='F'), dR, dZ)[1].flatten(order='F')
        for j in range(N):
            psum=0
            ffsum=0
            for i in range(pprime_order):
                if i==0:
                    pass
                else:
                    psum += c[i]*math.comb(i,1)*x[j]**(i-1)

            for i in range(ffprime_order):
                if i==0:
                    pass
                else:
                    ffsum += c[i+pprime_order]*math.comb(i,1)*x[j]**(i-1)
            B[j,nc] = x_z[j] * (R[j]*psum + 1/(mu0*R[j])*ffsum)

    else:
        B = np.zeros((N, nc))
        # p' Coefficients
        for i in range(pprime_order):
            for j in range(N):
                B[j, i] = R[j] * x[j] ** i

        # ff' relations
        for i in range(ffprime_order):
            for j in range(N):
                B[j, i + pprime_order] = (1/ (mu0 * R[j])) * x[j] ** i
    return B


# Finding total operator matrix E (nm+n_coils + nc+n_coils+1)
def get_E(A,Gc, Gvessel=None):
    B = np.identity(Gc.shape[1])
    C = np.zeros((Gc.shape[1], A.shape[1]))
    if Gvessel is not None:
        D = np.zeros((Gc.shape[1], Gvessel.shape[1]))
        E = np.block([[A,Gc,Gvessel],[C,B,D]])
    else:
        E = np.block([[A,Gc],[C,B]])

    return E


# Coefficient Matrix c (nc x 1)
def get_c(A, M):
    """
    Calculating coefficients

    Parameters
    ----------
    A - operator matrix
    M - measurements

    Returns
    -------
    c - coefficient matrix
    """
    return np.matmul(get_A_inv(A), M)


# Calculate a 2 dimensional normalised psi
def get_x(eq, jtor=None, psi_bndry=None, check_limited=False):
    """
    Function for calling elliptical solver, determining new psi then finding mask and normalising

    Parameters
    ----------
    eq - equilibrium object
    jtor - 2d current density matrix
    psi_bndry - a predetermined value of psi on the boundary

    Returns
    -------

    """

    eq.check_limited = check_limited
    eq._updateBoundaryPsi()

    if (jtor is not None):
        from scipy.integrate import trapz
        eq.Jtor = jtor  # update eq jtor attirbute

        # Using Elliptical Solver to calculate plasma_psi from plasma jtor profile
        rhs = -mu0 * eq.R * jtor

        # dont use plasma psi, calculate expected from new jtor and greens
        R = eq.R
        Z = eq.Z
        nx, ny = rhs.shape

        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]

        # List of indices on the boundary
        bndry_indices = np.concatenate(
            [   [(x, 0) for x in range(nx)],
                [(x, ny - 1) for x in range(nx)],
                [(0, y) for y in range(ny)],
                [(nx - 1, y) for y in range(ny)]])

        for x, y in bndry_indices:
            # Calculate the response of the boundary point
            # to each cell in the plasma domain
            greenfunc = Greens(R, Z, R[x, y], Z[x, y])

            # Prevent infinity/nan by removing (x,y) point
            greenfunc[x, y] = 0.0

            # Integrate over the domain
            rhs[x, y] = trapz(trapz(greenfunc * jtor)) * dR * dZ
            eq.plasma_psi[x, y] = rhs[x,y]

        plasma_psi = eq._solver(eq.plasma_psi, rhs)
        eq._updatePlasmaPsi(plasma_psi)

    # Fetch total psi (plasma + coils)
    psi = eq.psi()

    # Calculating Locations for axis and boundaries
    opt, xpt = critical.find_critical(eq.R, eq.Z, psi)
    psi_axis = opt[0][2]


    # we never use any mask, find out where it should go
    if psi_bndry is not None:
        mask = critical.core_mask(eq.R, eq.Z, psi, opt, xpt, psi_bndry)
    elif xpt:
        psi_bndry = xpt[0][2]
        mask = critical.core_mask(eq.R, eq.Z, psi, opt, xpt)
    else:
        # No X-points
        psi_bndry = psi[0, 0]
        mask = None

    # Calculate normalised psi.
    # 0 = magnetic axis
    # 1 = plasma boundary

    psi_norm = np.clip(((psi - psi_axis) / (psi_bndry - psi_axis)), 0, 1)

    return psi_norm, mask


# Calculate te fitting weight vector
def get_F(sigma):
    """
    Uses the inputted sigma vector to create the diagonal fitted weight matrix

    Parameters
    ----------
    sigma - vector containing measurement uncertainties

    """

    Flist = []
    for val in sigma:
        Flist.append(val[0]**(-1))
    F=np.diag(Flist)
    return F


#Calculate T
def get_T(eq, m, n, inside=True):
    """

    Parameters
    ----------
    eq - equilibrium object
    m - number of radial finite elements
    n - number of vertical finite elements
    inside - option to include finite elements on boundary

    Returns
    -------
    T - basis matrix for current initialisation
    """
    m+=2
    n+=2
    R = eq.R.flatten(order='F')
    Z = eq.Z.flatten(order='F')
    a, b, c, d = eq.Rmin, eq.Rmax, eq.Zmin, eq.Zmax
    dR = (b - a) / (m-1)
    dZ = (d - c) / (n-1)
    r_FE = np.linspace(a, b, m)
    z_FE = np.linspace(c, d, n)

    if inside:
        T = np.zeros((eq.nx * eq.ny, (m - 2) * (n - 2)))
        row_num=0
        for i in range(1,m-1):
            r_h = r_FE[i]
            for j in range(1,n-1):
                z_h = z_FE[j]

                for h in range(T.shape[0]):
                    if (abs(R[h]-r_h)/dR)<1:
                        if (abs(Z[h]-z_h)/dZ<1):
                            T[h,row_num] = (1-abs(R[h]-r_h)/dR)*(1-abs(Z[h]-z_h)/dZ)
                row_num +=1

    else:
        T = np.zeros((eq.nx * eq.ny,m*n))
        row_num = 0
        for i in range(m):
            r_h = r_FE[i]
            for j in range(n):
                z_h = z_FE[j]

                for h in range(T.shape[0]):
                    if (abs(R[h] - r_h) / dR) < 1:
                        if (abs(Z[h] - z_h) / dZ < 1):
                            T[h, row_num] = (1 - abs(R[h] - r_h) / dR) * (
                                        1 - abs(Z[h] - z_h) / dZ)
                row_num += 1
    return T
