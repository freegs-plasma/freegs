import numpy as np
import scipy
import math
from . import critical, plotting, jtor, control, picard, boundary, equilibrium, filament_coil, machine
from .machine import Coil, ShapedCoil, FilamentCoil, RogowskiSensor, PoloidalFieldSensor, FluxLoopSensor, Filament, Filament_Group
from .gradshafranov import Greens, GreensBr, GreensBz
from shapely.geometry import Point, Polygon, LineString, LinearRing
from scipy.special import ellipk, ellipe
from scipy.constants import mu_0

class Reconstruction:
    def __init__(self, tokamak, pprime_order, ffprime_order, eq=None,tolerance=1e-7, VesselCurrents=True, VerticalControl=True, CurrentInitialisation=True, FittingWeight=True, check_limited=True, returnChi=False, pause=0.01, show=True):
        self.tokamak = tokamak
        self.eq = eq
        self.pprime_order = pprime_order
        self.ffprime_order = ffprime_order

        self.VesselCurrents = VesselCurrents
        self.VerticalControl = VerticalControl
        self.FittingWeight = FittingWeight
        self.CurrentInitialisation = CurrentInitialisation
        self.show=show
        self.check_limited = check_limited


        # Reconstruction attributes
        self.x = 0 # noramlised psi
        # self.Gplasma = self.get_PlasmaGreens()
        # self.Gcoil = self.get_CoilGreens()
        # self.Gfil = self.get_FilamentGreens()


    # Creating an equilibrium
    def generate_Measurements(self, alpha_m, alpha_n, Rmin=0.1, Rmax=2, Zmin=-1,Zmax=1, nx=65, ny=65, x_z1=0.6, x_z2=-0.6, x_r1=1.1, x_r2=1.1):
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

        eq = equilibrium.Equilibrium(tokamak=self.tokamak,
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
                     show=self.show,
                     check_limited=self.check_limited)

        print('Construction Diagnostics')
        self.tokamak.printMeasurements(equilibrium=eq)
        self.tokamak.printCurrents()
        self.eq = eq

    #Taking measurements from tokamak
    def take_Measurements_from_tokamak(self):
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

        for sensor in self.tokamak.sensors:
            M.append([sensor.measurement])
            if sensor.measurement == 0:
                sigma.append([1])
            else:
                sigma.append([sensor.measurement / sensor.weight])

        # Adding Coil Currents To Machine and M
        for name, coil in self.tokamak.coils:
            M.append([coil.current])
            sigma.append([1e-5])

        self.M = M
        self.sigma = sigma

    #Perform a Chi Squared test
    def chi_squared_test(self):
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
        for i in range(len(self.M)):
            cs += ((self.M[i] - self.H[i]) / self.sigma[i]) ** 2
        return cs

    # Calculating the plasma contirbution to the measurements
    def get_M_plasma(self):
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
        self.tokamak.takeMeasurements()
        for i in range(len(self.M)-len(self.tokamak.coils)):
            sensor = self.tokamak.sensors[i]
            M_plasma.append([self.M[i][0] - sensor.measurement])
        return M_plasma

    # Calculate the Current initialisation T matrix
    def current_initialise(self, T):
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

        M_plasma = self.get_M_plasma()

        if not isinstance(T, np.ndarray):
            T = get_T(eq, 5, 5)

        coefs = scipy.linalg.lstsq(self.Gplasma @ T, M_plasma)[0]
        jtor = T @ coefs
        return jtor

    def get_PlasmaGreens(self):
        """
        Function for calculating greens matrix for the plasma response to machine sensors.
        Runs on initialisation only

        Parameters
        ----------
        self.eq - self.equilibrium object

        Returns
        -------
        PlasmaGreens - Plasma Greens Matrix
        """

        # Creating Grid
        N = self.eq.nx * self.eq.ny
        PlasmaGreens = np.zeros((len(self.tokamak.sensors), N))
        dR = self.eq.R[1, 0] - self.eq.R[0, 0]
        dZ = self.eq.Z[0, 1] - self.eq.Z[0, 0]

        for w, sensor in enumerate(self.tokamak.sensors):
            dim = self.eq.R.shape
            # Creating 2d array over self.equilibrium Points
            greens_matrix = np.zeros(dim)

            # If Rog sensors, find all points on self.equilibrium grid that lie inside rog
            # Greens function is then simply 1 (or dRdZ with the integral) as the rog measures current
            if isinstance(sensor, RogowskiSensor):
                sensor_loc = sensor.to_numpy()
                Rs = sensor_loc[0]
                Zs = sensor_loc[1]
                polygon_list = []

                for r, z in zip(Rs, Zs):
                    point = (r, z)
                    polygon_list.append(point)

                polygon = Polygon(polygon_list)

                for i in range(dim[0]):
                    for j in range(dim[1]):
                        if polygon.contains(Point(self.eq.R[i, j], self.eq.Z[i, j])):
                            greens_matrix[i, j] = dR * dZ

            # If FL sensor, find position of flux loop sensor and then find the greens function relating unit current at each grid position to psi at sensor
            if isinstance(sensor, FluxLoopSensor):
                for i in range(dim[0]):
                    for j in range(dim[1]):
                        greens_matrix[i, j] = dR * dZ * Greens(self.eq.R[i, j],
                                                               self.eq.Z[i, j],
                                                               sensor.R,
                                                               sensor.Z)

            # If BP sensor, find position of sensor and then find the greens function relating unit current at each grid position to B field
            if isinstance(sensor, PoloidalFieldSensor):
                for i in range(dim[0]):
                    for j in range(dim[1]):
                        greens_matrix[i, j] = dR * dZ * (
                                GreensBr(self.eq.R[i, j], self.eq.Z[i, j], sensor.R,
                                         sensor.Z) * np.cos(
                            sensor.theta) + GreensBz(self.eq.R[i, j],
                                                     self.eq.Z[i, j], sensor.R,
                                                     sensor.Z) * np.sin(
                            sensor.theta))

            # Convert the grid to a line, then append the greens plasma response matrix with it
            greens_row = greens_matrix.flatten(order='F')
            PlasmaGreens[w] = greens_row
        self.Gplasma = PlasmaGreens

    def get_CoilGreens(self):
        n_coils = len(self.tokamak.coils)
        CoilGreens = np.zeros((len(self.tokamak.sensors), n_coils))
        dR = self.eq.R[1, 0] - self.eq.R[0, 0]
        dZ = self.eq.Z[0, 1] - self.eq.Z[0, 0]

        for w, sensor in enumerate(self.tokamak.sensors):
            for coil_num, coil in enumerate(self.tokamak.coils):
                coil = coil[1]  # Getting the coil

                if isinstance(sensor, RogowskiSensor):
                    sensor_loc = sensor.to_numpy()
                    Rs = sensor_loc[0]
                    Zs = sensor_loc[1]
                    polygon_list = []

                    for r, z in zip(Rs, Zs):
                        point = (r, z)
                        polygon_list.append(point)

                    # Create rog sensor polygon
                    polygon = Polygon(polygon_list)

                    # If shaped coil find the proportion of shaped coil inside rog
                    if isinstance(coil, ShapedCoil):
                        Shaped_Coil_List = []
                        for shape in coil.shape:
                            Shaped_Coil_List.append(Point(shape))
                        Shaped_Coil = Polygon(Shaped_Coil_List)
                        CoilGreens[w, coil_num] = (polygon.intersection(
                            Shaped_Coil).area) / (coil._area)

                    # If coil, find if coil lies inside rog
                    elif isinstance(coil, Coil):
                        point = Point(coil.R, coil.Z)
                        if polygon.contains(point):
                            CoilGreens[w, coil_num] = 1

                    # If coil is a circuit, find coils inside circuit
                    elif isinstance(coil, Circuit):
                        for name, sub_coil, multiplier in coil.coils:

                            # If itis a filament coil, loop through all the wires in the filament coil
                            # If the wire lies inside, append greens matrix by 1, as the coefficient that is being modeleld for the circuit current is the current in each wire aswell
                            if isinstance(sub_coil, FilamentCoil):
                                for r, z in sub_coil.points:
                                    point = Point(r, z)
                                    if polygon.contains(point):
                                        CoilGreens[w, coil_num] += 1

                # Using the greens functions already specificed, calculate the effect of a unit current through the coil at the position of the sensor
                if isinstance(sensor, FluxLoopSensor):
                    CoilGreens[w, coil_num] += coil.controlPsi(sensor.R,
                                                               sensor.Z)

                if isinstance(sensor, PoloidalFieldSensor):
                    CoilGreens[w, coil_num] += np.cos(
                        sensor.theta) * coil.controlBr(sensor.R,
                                                       sensor.Z) + np.sin(
                        sensor.theta) * coil.controlBz(sensor.R, sensor.Z)

        self.Gcoil = CoilGreens

    def get_FilamentGreens(self):
        n_fils = 0
        for fil in self.tokamak.vessel:
            if isinstance(fil, Filament):
                n_fils += 1
            if isinstance(fil, Filament_Group):
                n_fils += len(fil.filaments)

        FilamentGreens = np.zeros((len(self.tokamak.sensors), n_fils))
        dR = self.eq.R[1, 0] - self.eq.R[0, 0]
        dZ = self.eq.Z[0, 1] - self.eq.Z[0, 0]

        # Works very similarly to coils, loops through each of the sensors and each of the filaments, find the effect of unit current through each filament on the measurement
        for w, sensor in enumerate(self.tokamak.sensors):
            fil_num = 0
            for fil in self.tokamak.vessel:
                if isinstance(fil, Filament):
                    if isinstance(sensor, RogowskiSensor):
                        sensor_loc = sensor.to_numpy()
                        Rs = sensor_loc[0]
                        Zs = sensor_loc[1]
                        polygon_list = []

                        for r, z in zip(Rs, Zs):
                            point = (r, z)
                            polygon_list.append(point)

                        polygon = Polygon(polygon_list)

                        point = Point(fil.R, fil.Z)
                        if polygon.contains(point):
                            FilamentGreens[w, fil_num] = 1

                    if isinstance(sensor, FluxLoopSensor):
                        FilamentGreens[w, fil_num] = fil.controlPsi(sensor.R,
                                                                    sensor.Z)

                    if isinstance(sensor, PoloidalFieldSensor):
                        FilamentGreens[w, fil_num] = np.cos(
                            sensor.theta) * fil.controlBr(
                            sensor.R, sensor.Z) + np.sin(
                            sensor.theta) * fil.controlBz(sensor.R, sensor.Z)

                    fil_num += 1

                if isinstance(fil, Filament_Group):
                    for sub_fil in fil.filaments:

                        if isinstance(sensor, RogowskiSensor):
                            sensor_loc = sensor.to_numpy()
                            Rs = sensor_loc[0]
                            Zs = sensor_loc[1]
                            polygon_list = []

                            for r, z in zip(Rs, Zs):
                                point = (r, z)
                                polygon_list.append(point)

                            polygon = Polygon(polygon_list)

                            point = Point(sub_fil.R, sub_fil.Z)
                            if polygon.contains(point):
                                FilamentGreens[w, fil_num] += 1

                        if isinstance(sensor, FluxLoopSensor):
                            FilamentGreens[w, fil_num] += sub_fil.controlPsi(
                                sensor.R, sensor.Z)

                        if isinstance(sensor, PoloidalFieldSensor):
                            FilamentGreens[w, fil_num] += np.cos(
                                sensor.theta) * sub_fil.controlBr(
                                sensor.R, sensor.Z) + np.sin(
                                sensor.theta) * sub_fil.controlBz(sensor.R,
                                                                  sensor.Z)

                        fil_num += 1
        self.Gfil = FilamentGreens

    # Basis Matrix B (N x nc)
    def get_B(self, c=None, VC=False):
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
        N = self.eq.nx * self.eq.ny
        R = self.eq.R.flatten(order='F')
        nc = self.pprime_order + self.ffprime_order
        dR = self.eq.R[1, 0] - self.eq.R[0, 0]
        dZ = self.eq.Z[0, 1] - self.eq.Z[0, 0]

        if VC and self.VerticalControl:
            B = np.zeros((N, nc + 1))
            for i in range(self.pprime_order):
                for j in range(N):
                    B[j, i] = R[j] * (self.x[j]) ** i

            # ff' relations
            for i in range(self.ffprime_order):
                for j in range(N):
                    B[j, i + self.pprime_order] = (1 / (mu_0 * R[j])) * (self.x[j]) ** i

            x_z = np.gradient(np.reshape(self.x, (self.eq.nx, self.eq.ny), order='F'), dR, dZ)[1].flatten(order='F')
            for j in range(N):
                psum = 0
                ffsum = 0
                for i in range(self.pprime_order):
                    if i == 0:
                        pass
                    else:
                        psum += c[i] * math.comb(i, 1) * self.x[j] ** (i - 1)

                for i in range(self.ffprime_order):
                    if i == 0:
                        pass
                    else:
                        ffsum += c[i + self.pprime_order] * math.comb(i, 1) * self.x[
                            j] ** (i - 1)
                B[j, nc] = x_z[j] * (R[j] * psum + 1 / (mu_0 * R[j]) * ffsum)

        else:
            B = np.zeros((N, nc))
            # p' Coefficients
            for i in range(self.pprime_order):
                for j in range(N):
                    B[j, i] = R[j] * self.x[j] ** i

            # ff' relations
            for i in range(self.ffprime_order):
                for j in range(N):
                    B[j, i + self.pprime_order] = (1 / (mu_0 * R[j])) * self.x[j] ** i

        return B

    # Finding total operator matrix E (nm+n_coils + nc+n_coils+1)
    def get_E(self, A):
        B = np.identity(self.Gcoil.shape[1])
        C = np.zeros((self.Gcoil.shape[1], A.shape[1]))
        if self.Gvessel is not None:
            D = np.zeros((self.Gcoil.shape[1], self.Gvessel.shape[1]))
            E = np.block([[A, self.Gcoil, self.Gvessel], [C, B, D]])
        else:
            E = np.block([[A, self.Gcoil], [C, B]])

        return E

    # Calculate a 2 dimensional normalised psi
    def get_x(self, jtor):
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

        self.eq.check_limited = self.check_limited
        self.eq._updateBoundaryPsi()

        if jtor is not None:
            from scipy.integrate import trapz
            self.eq.Jtor = jtor  # update eq jtor attirbute

            # Using Elliptical Solver to calculate plasma_psi from plasma jtor profile
            rhs = -mu_0 * self.eq.R * jtor

            # dont use plasma psi, calculate expected from new jtor and greens
            R = self.eq.R
            Z = self.eq.Z
            nx, ny = rhs.shape

            dR = R[1, 0] - R[0, 0]
            dZ = Z[0, 1] - Z[0, 0]

            # List of indices on the boundary
            bndry_indices = np.concatenate(
                [[(x, 0) for x in range(nx)],
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
                self.eq.plasma_psi[x, y] = rhs[x, y]

            plasma_psi = self.eq._solver(self.eq.plasma_psi, rhs)
            self.eq._updatePlasmaPsi(plasma_psi)

        # Fetch total psi (plasma + coils)
        psi = self.eq.psi()

        # Calculating Locations for axis and boundaries
        opt, xpt = critical.find_critical(self.eq.R, self.eq.Z, psi)
        psi_axis = opt[0][2]

        if xpt:
            psi_bndry = xpt[0][2]
            mask = critical.core_mask(self.eq.R, self.eq.Z, psi, opt, xpt)
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
    def get_F(self):
        """
        Uses the inputted sigma vector to create the diagonal fitted weight matrix

        Parameters
        ----------
        sigma - vector containing measurement uncertainties

        """

        Flist = []
        for val in self.sigma:
            Flist.append(val[0] ** (-1))
        F = np.diag(Flist)
        return F

    # Calculate T
    def get_T(self, m, n):
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
        m += 2
        n += 2
        R = self.eq.R.flatten(order='F')
        Z = self.eq.Z.flatten(order='F')
        a, b, c, d = self.eq.Rmin, self.eq.Rmax, self.eq.Zmin, self.eq.Zmax
        dR = (b - a) / (m - 1)
        dZ = (d - c) / (n - 1)
        r_FE = np.linspace(a, b, m)
        z_FE = np.linspace(c, d, n)

        T = np.zeros((self.eq.nx * self.eq.ny, (m - 2) * (n - 2)))
        row_num = 0
        for i in range(1, m - 1):
            r_h = r_FE[i]
            for j in range(1, n - 1):
                z_h = z_FE[j]

                for h in range(T.shape[0]):
                    if (abs(R[h] - r_h) / dR) < 1:
                        if (abs(Z[h] - z_h) / dZ < 1):
                            T[h, row_num] = (1 - abs(R[h] - r_h) / dR) * (
                                        1 - abs(Z[h] - z_h) / dZ)
                row_num += 1

        return T

    def generate_Greens(self):
        self.get_PlasmaGreens()
        self.get_CoilGreens()
        self.get_FilamentGreens()

    def solve(self, pause=0.01, tolerance=1e-7):
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
        VerticalControl - option for ferron's vertical control
        CI - option for current initialisation
        FittingWeight - option to apply a fitting weight vector to scale lstsq calculation
        returnChi - option to return the final value of chi squared

        """

        if self.show:
            import matplotlib.pyplot as plt
            from .plotting import plotEquilibrium

            if pause > 0.0:
                # No axis specified, so create a new figure
                fig = plt.figure()
                axis = fig.add_subplot(111)

        if self.VesselCurrents:
            self.Gvessel = self.Gfil @ self.tokamak.eigenbasis

        jtor_2d = None

        # Performs plasma current density initialisation
        if self.CurrentInitialisation:
            T = self.get_T(5, 5)
            jtor_1d = self.current_initialise(T)
            jtor_2d = np.reshape(jtor_1d, (self.eq.nx, self.eq.ny), order='F')

        # Fetching our initial normalised psi
        x_2d, mask = self.get_x(jtor_2d)
        self.x = x_2d.flatten('F')

        # Calculate B and apply mask
        B = self.get_B()
        if mask is not None:
            maskline = mask.flatten('F')
            for j in range(B.shape[1]):
                for i in range(B.shape[0]):
                    B[i, j] *= maskline[i]

        # Calculating operator matrix A
        A = self.Gplasma @ B

        if self.VesselCurrents:
            E = self.get_E(A)
        else:
            E = self.get_E(A)

        # Performing least squares calculation for c
        if self.FittingWeight:
            F = self.get_F()
            c = scipy.linalg.lstsq(F @ E, F @ self.M)[0]
        else:
            c = scipy.linalg.lstsq(E, self.M)[0]

        # Computed Measurement and Chi Squred Test
        self.H = np.matmul(E, c)
        chi_squared = self.chi_squared_test()

        # start iterative loop
        it = 0  # iteration number
        while True:
            if self.show:
                # Plot state of plasma equilibrium
                if pause < 0:
                    fig = plt.figure()
                    axis = fig.add_subplot(111)
                else:
                    axis.clear()

                plotEquilibrium(self.eq, axis=axis, show=False)

                if pause < 0:
                    # Wait for user to close the window
                    plt.show()
                else:
                    # Update the canvas and pause
                    # Note, a short pause is needed to force drawing update
                    axis.figure.canvas.draw()
                    plt.pause(pause)

            # Use B & C to calculate Jtor matrix
            if self.VesselCurrents:
                jtor_1d = B @ c[:-(len(self.tokamak.coils) + self.Gvessel.shape[1])]
            else:
                jtor_1d = B @ c[:-(len(self.tokamak.coils))]

            jtor_2d = np.reshape(jtor_1d, (self.eq.nx, self.eq.ny), order='F')

            # Recalculate Psi values with elliptical solver
            x_2d, mask = self.get_x(jtor_2d)
            self.x = x_2d.flatten('F')

            # Recalculate B and A matrices from new Psi
            B = self.get_B(c=c, VC=True)

            if mask is not None:
                maskline = mask.flatten('F')
                for j in range(B.shape[1]):
                    for i in range(B.shape[0]):
                        B[i, j] *= maskline[i]

            # Calculating operator matrix A
            A = self.Gplasma @ B
            if self.VesselCurrents:
                E = self.get_E(A)
            else:
                E = self.get_E(A)

            # Performing least squares calculation for c
            if self.FittingWeight:
                c = scipy.linalg.lstsq(F @ E, F @ self.M)[0]
            else:
                c = scipy.linalg.lstsq(E, self.M)[0]

            if self.VesselCurrents:
                fil_currents = self.tokamak.eigenbasis @ c[-(
                self.tokamak.eigenbasis.shape[1]):]
                self.tokamak.updateVesselCurrents(fil_currents)
            for i in range(len(self.tokamak.coils)):
                circuit = self.tokamak.coils[i][1]
                circuit.current = c[self.pprime_order + self.ffprime_order + 1 + i][0]

            # Take Diagnostics and Perform Convergence Test
            self.H = E @ c
            chi_last = chi_squared
            chi_squared = self.chi_squared_test()
            print('Chi Squared = ', chi_squared)
            it += 1

            if chi_squared <= tolerance or float('%.4g' % chi_squared) == float('%.4g' % chi_last) or it > 50:
                if it > 50:
                    print('Finished due to runtime')
                break

        print(it)
        if self.VesselCurrents:
            print('dz=', c[
                -len(self.tokamak.coils) - 1 - self.Gvessel.shape[1]])
        else:
            print('dz=', c[-len(self.tokamak.coils) - 1])

        self.c = c

        print(' ')
        print('Coefficients (alpha_i, dz, coil currents, vessel basis coefficients')
        print(c)

        # Printing final constructed values
        print(' ')
        print('Reconstruction Sensor Diagnostics')
        self.tokamak.printMeasurements(equilibrium=self.eq)
        self.tokamak.printCurrents()

        for sensor, val1, val2 in zip(self.tokamak.sensors, self.M, self.H):
            print(sensor.name, val1, val2, sensor.measurement)

