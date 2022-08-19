import numpy as np
import scipy
from .recon_tools import grid_to_line, line_to_grid, chi_squared_test, blender, print_H, current_initialise
from .recon_matrices import get_E, get_B, get_c, get_x, get_A, get_F, get_T
from . import plotting


# Perform the iteration
def solve(tokamak, eq, M, sigma, pprime_order, ffprime_order, tolerance=1, blend=0,
          show=True, pause=0.01, axis=None, VC=False, CI=False, Fscale=True, VesselCurrents=False, returnChi=False, check_limited=False):

    """
    Parameters
    ----------
    tokamak - tokamak object
    eq - equilibrium object
    M - measurements
    sigma - measurement uncertainties
    pprime_order - number of polynomial coefficients for pprime model
    ffprime_order - number of polynomial coefficients for ffprime model
    tolerance - value beneath which the chi squared value will stop the iteration
    G - Greens matrix
    Gdz - vertical control greens matrix
    blend - blending coefficient
    show - option to display psi through the iterations
    pause - time to wait between iterations
    axis - axis to plot iterations on
    VC - option for ferron's vertical control
    CI - option for current initialisation
    Fscale - option to apply a fitting weight vector to scale lstsq calculation
    returnChi - option to return the final value of chi squared

    """

    if show:
        import matplotlib.pyplot as plt
        from .plotting import plotEquilibrium

        if pause > 0.0 and axis is None:
            # No axis specified, so create a new figure
            fig = plt.figure()
            axis = fig.add_subplot(111)


    if not hasattr(tokamak, 'Gplasma'):
        tokamak.get_PlasmaGreens(eq)
    if not hasattr(tokamak, 'Gcoil'):
        tokamak.get_CoilGreens(eq)
    if VesselCurrents:
        if not hasattr(tokamak, 'Gfil'):
            tokamak.get_FilamentGreens(eq)
        tokamak.Gvessel = tokamak.Gfil @ tokamak.eigenbasis

    jtor_2d = None

    # Performs plasma current density initialisation
    if CI:
        T = get_T(eq,5,5)
        jtor_1d = current_initialise(M[:-len(tokamak.coils)],tokamak, eq, T=T)
        jtor_2d = line_to_grid(jtor_1d, eq.nx, eq.ny)

    # Fetching our initial normalised psi
    x_2d, mask = get_x(eq,jtor=jtor_2d, check_limited=check_limited)
    x_1d = grid_to_line(x_2d)


    #Calculate B and apply mask
    B = get_B(x_1d, eq, pprime_order, ffprime_order)
    if mask is not None:
        maskline = grid_to_line(mask)
        for j in range(B.shape[1]):
            for i in range(B.shape[0]):
                B[i, j] *= maskline[i]


    # Calculating operator matrix A
    A = tokamak.Gplasma @ B

    if VesselCurrents:
        E = get_E(A,tokamak.Gcoil, tokamak.Gvessel)
    else:
        E= get_E(A,tokamak.Gcoil)

    # Performing least squares calculation for c
    if Fscale:
        F = get_F(sigma)
        c = scipy.linalg.lstsq(F@E, F@M)[0]
    else:
        c = scipy.linalg.lstsq(E, M)[0]


    # Computed Measurement and Chi Squred Test
    H = np.matmul(E,c)
    chi_squared = chi_squared_test(M,sigma,H)


    #start iterative loop
    it=0 # iteration number
    while True:
        if show:
            # Plot state of plasma equilibrium
            if pause < 0:
                fig = plt.figure()
                axis = fig.add_subplot(111)
            else:
                axis.clear()

            plotEquilibrium(eq, axis=axis, show=False)

            if pause < 0:
                # Wait for user to close the window
                plt.show()
            else:
                # Update the canvas and pause
                # Note, a short pause is needed to force drawing update
                axis.figure.canvas.draw()
                plt.pause(pause)


        # Use B & C to calculate Jtor matrix
        if VesselCurrents:
            jtor_1d = np.matmul(B, c[:-(len(tokamak.coils)+tokamak.Gvessel.shape[1])])
        else:
            jtor_1d = np.matmul(B, c[:-(len(tokamak.coils))])

        jtor_2d = line_to_grid(jtor_1d, eq.nx, eq.ny)


        #Recalculate Psi values with elliptical solver
        x_last = x_1d
        x_2d, mask = get_x(eq, jtor=jtor_2d, check_limited=check_limited)
        x_1d = blender(grid_to_line(x_2d),x_last,blend)

        # Recalculate B and A matrices from new Psi
        B = get_B(x_1d, eq, pprime_order, ffprime_order, c=c, VC=VC)
        if mask is not None:
            maskline = grid_to_line(mask)
            for j in range(B.shape[1]):
                for i in range(B.shape[0]):
                    B[i, j] *= maskline[i]


        # Calculating operator matrix A
        A = get_A(tokamak.Gplasma, B)
        if VesselCurrents:
            E = get_E(A, tokamak.Gcoil, tokamak.Gvessel)
        else:
            E = get_E(A, tokamak.Gcoil)

        # Performing least squares calculation for c
        if Fscale:
            c = scipy.linalg.lstsq(F @ E, F @ M)[0]
        else:
            c = scipy.linalg.lstsq(E, M)[0]

        if VesselCurrents:
            fil_currents = tokamak.eigenbasis @ c[-(tokamak.eigenbasis.shape[1]):]
            tokamak.updateVesselCurrents(fil_currents)

        for i in range(tokamak.Gcoil.shape[1]):
            name, circuit = tokamak.coils[i]
            circuit.current = c[pprime_order+ffprime_order+1+i][0]

        #Take Diagnostics and Perform Convergence Test
        H=E@c
        chi_last = chi_squared
        chi_squared = chi_squared_test(M, sigma, H)
        print('Chi Squared = ', chi_squared)
        it+=1

        if chi_squared <= tolerance or float('%.4g' % chi_squared) == float('%.4g' % chi_last) or it>50:
            if it>50:
                print('Finished due to runtime')
            if VesselCurrents:
                print('dz=',c[-len(tokamak.coils)-1-tokamak.Gvessel.shape[1]])
            else:
                print('dz=', c[-len(tokamak.coils) - 1])
            print(it)
            break

    print(' ')
    print('Coefficients (alpha_i, dz, coil currents, vessel basis coefficients')
    print(c)


    # Printing final constructed values
    print(' ')
    print('Reconstruction Sensor Diagnostics')
    tokamak.printMeasurements(equilibrium=eq)
    tokamak.printCurrents()

    for sensor, val1, val2 in zip(tokamak.sensors, M,H):
        print(sensor.name, val1, val2, sensor.measurement)


    if returnChi:
        return chi_squared[0], eq, c