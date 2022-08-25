import numpy as np
import scipy
import math
from . import critical, plotting, jtor, control, picard, boundary, filament_coil, machine
from .equilibrium import Equilibrium
from .machine import Coil, ShapedCoil, FilamentCoil, RogowskiSensor, PoloidalFieldSensor, FluxLoopSensor, Filament, Filament_Group, Wall, Circuit, Machine
from .gradshafranov import Greens, GreensBr, GreensBz
from shapely.geometry import Point, Polygon, LineString, LinearRing
from scipy.special import ellipk, ellipe
from scipy.constants import mu_0
import matplotlib.pyplot as plt

class Reconstruction(Equilibrium):
    def __init__(self, tokamak, pprime_order, ffprime_order,  nx=65, ny=65, Rmin=0.1, Rmax=2, Zmin=-1, Zmax=1, tolerance=1e-7, VesselCurrents=True, VerticalControl=True, CurrentInitialisation=True, show=True, test=False):
        Equilibrium.__init__(self, tokamak=tokamak, Rmin=Rmin, Rmax=Rmax, Zmin=Zmin, Zmax=Zmax, nx=nx, ny=ny)
        self.test=test

        # Defining Reconstruction Parameters
        self.pprime_order = pprime_order
        self.ffprime_order = ffprime_order
        self.VesselCurrents = VesselCurrents
        self.VerticalControl = VerticalControl
        self.CurrentInitialisation = CurrentInitialisation
        self.tolerance = tolerance
        self.show=show

        # Creating Reconstruction Grid and Greens functions
        self.generate_Greens()

        # Reconstruction attributes
        self.x = None # normalised psi
        self.c = None # coefficients

    # Methods for building measurment and uncertainty matrices
    def take_Measurements_from_dictionary(self, measurement_dict, sigma_dict):
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
            M.append([measurement_dict[sensor.name]])
            if sigma_dict is not None:
                self.FittingWeight = True
                sigma.append([sigma_dict[sensor.name]])
            else:
                self.FittingWeight = False
                sigma.append([1])

        for name, coil in self.tokamak.coils:
            M.append([measurement_dict[name]])
            sigma.append([sigma_dict[name]])

        self.M = M
        self.sigma = sigma

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
        self.FittingWeight=True

        for sensor in self.tokamak.sensors:
            M.append([sensor.measurement])
            if sensor.measurement == 0:
                sigma.append([1])
            else:
                sigma.append([1 / sensor.weight])

        # Adding Coil Currents To Machine and M
        for name, coil in self.tokamak.coils:
            M.append([coil.current])
            sigma.append([1/50])

        self.M = M
        self.sigma = sigma

    # Pass the coil current sto the equilibrium object
    def initialise_coil_current(self):
        """
        Method for giving the tokamak object the correct currents for reconstruction
        """
        for index, (name, coil) in enumerate(self.tokamak.coils):
            coil.current = self.M[-len(self.tokamak.coils) + index][0]

    # Calculate the Current initialisation T matrix
    def initialise_plasma_current(self, m=5, n=5):
        """
        Function for calculating the initial jtor

        Parameters
        ----------
        Gplasma - greens matrix
        M_plasma - plasma contribution to measurements
        T - Initialisation basis matrix

        Returns
        -------
        jtor - current density matrix
        """

        M_plasma = self.get_M_plasma()
        T = self.get_T(m=m,n=n)

        coefs = scipy.linalg.lstsq(self.Gplasma @ T, M_plasma)[0]
        jtor = T @ coefs
        return jtor

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
        for i in range(len(self.M) - len(self.tokamak.coils)):
            sensor = self.tokamak.sensors[i]
            M_plasma.append([self.M[i][0] - sensor.measurement])
        return M_plasma

    # Applying mask to a 2d matrix
    def apply_mask(self, Matrix, mask):
        if mask is not None:
            maskline = mask.flatten('F')
            for j in range(Matrix.shape[1]):
                for i in range(Matrix.shape[0]):
                    Matrix[i, j] *= maskline[i]
        return Matrix

    # Updating vessel and filament currents
    def update_currents(self):
        if self.VesselCurrents:
            fil_currents = self.tokamak.eigenbasis @ self.c[-(
            self.tokamak.eigenbasis.shape[1]):]
            self.tokamak.updateVesselCurrents(fil_currents)

        for i, (name, circuit) in enumerate(self.tokamak.coils):
            if self.it == 0 or not self.VerticalControl:
                circuit.current = self.c[self.pprime_order + self.ffprime_order + i][0]
            else:
                circuit.current = self.c[self.pprime_order + self.ffprime_order + 1 + i][0]

    #Perform a Chi Squared test
    def convergence_test(self, tolerance):
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

        new_chi = 0
        for i in range(len(self.M)):
            new_chi += ((self.M[i] - self.H[i]) / self.sigma[i]) ** 2

        print('Chi Squared =', new_chi)

        convergence = False

        if self.it == 0 and new_chi <= tolerance:
            convergence = True

        elif self.it>0 and (new_chi <= tolerance or float('%.3g' % new_chi) == float('%.3g' % self.chi)):
            convergence = True

        elif self.it>50:
            print('Finished due to runtime')
            convergence = True

        self.chi = new_chi
        return convergence

    # Calculating Greens functions
    def get_PlasmaGreens(self):
        """
        Function for calculating greens matrix for the plasma response to machine sensors.
        Runs on initialisation only

        Parameters
        ----------
        tokamak - location of sensors


        Returns
        -------
        PlasmaGreens - Plasma Greens Matrix
        """

        # Creating Grid
        N = self.nx * self.ny
        PlasmaGreens = np.zeros((len(self.tokamak.sensors), N))
        dR = self.R[1, 0] - self.R[0, 0]
        dZ = self.Z[0, 1] - self.Z[0, 0]

        for w, sensor in enumerate(self.tokamak.sensors):
            dim = self.R.shape
            # Creating 2d array over self.griduilibrium Points
            greens_matrix = np.zeros(dim)

            # If Rog sensors, find all points on self.griduilibrium grid that lie inside rog
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
                        if polygon.contains(Point(self.R[i, j], self.Z[i, j])):
                            greens_matrix[i, j] = dR * dZ

            # If FL sensor, find position of flux loop sensor and then find the greens function relating unit current at each grid position to psi at sensor
            if isinstance(sensor, FluxLoopSensor):
                for i in range(dim[0]):
                    for j in range(dim[1]):
                        greens_matrix[i, j] = dR * dZ * Greens(self.R[i, j],
                                                               self.Z[i, j],
                                                               sensor.R,
                                                               sensor.Z)

            # If BP sensor, find position of sensor and then find the greens function relating unit current at each grid position to B field
            if isinstance(sensor, PoloidalFieldSensor):
                for i in range(dim[0]):
                    for j in range(dim[1]):
                        greens_matrix[i, j] = dR * dZ * (
                                GreensBr(self.R[i, j], self.Z[i, j], sensor.R,
                                         sensor.Z) * np.cos(
                            sensor.theta) + GreensBz(self.R[i, j],
                                                     self.Z[i, j], sensor.R,
                                                     sensor.Z) * np.sin(
                            sensor.theta))

            # Convert the grid to a line, then append the greens plasma response matrix with it
            greens_row = greens_matrix.flatten(order='F')
            PlasmaGreens[w] = greens_row
        self.Gplasma = PlasmaGreens

    def get_CoilGreens(self):
        """
        Calculating the Coil Greens Matrix

        Parameters
        ------
        tokamak - machine with location of coils


        Returns
        -------
        Gcoil - Coil response matrix

        """
        n_coils = len(self.tokamak.coils)
        CoilGreens = np.zeros((len(self.tokamak.sensors), n_coils))
        dR = self.R[1, 0] - self.R[0, 0]
        dZ = self.Z[0, 1] - self.Z[0, 0]

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
        """
        Calculating the Filaments Greens Matrix

        Parameters
        ------
        tokamak - machine with location of filaments


        Returns
        -------
        Gfil - filament response matrix

        """
        n_fils = 0
        for fil in self.tokamak.vessel:
            if isinstance(fil, Filament):
                n_fils += 1
            if isinstance(fil, Filament_Group):
                n_fils += len(fil.filaments)

        FilamentGreens = np.zeros((len(self.tokamak.sensors), n_fils))
        dR = self.R[1, 0] - self.R[0, 0]
        dZ = self.Z[0, 1] - self.Z[0, 0]

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

    def generate_Greens(self):
        """
        Method for calling the greens function methods
        """
        self.get_PlasmaGreens()
        self.get_CoilGreens()
        if self.VesselCurrents:
            self.get_FilamentGreens()

    # Basis Matrix B (N x nc)
    def get_B(self):
        """
        Function for calculating the basis matrix, runs every iteration

        Parameters
        ----------
        x - normalised psi
        pprime_order - number of polynomial coefficients for pprime model
        ffprime_order - number of polynomial coefficients for ffprime model
        c - coefficients matrix

        Returns
        -------
        B - Basis matrix
        """
        # Create Empty Matrix
        N = self.nx * self.ny
        R = self.R.flatten(order='F')
        nc = self.pprime_order + self.ffprime_order
        dR = self.R[1, 0] - self.R[0, 0]
        dZ = self.Z[0, 1] - self.Z[0, 0]

        if self.VerticalControl and self.c is not None:
            B = np.zeros((N, nc + 1))
            for i in range(self.pprime_order):
                for j in range(N):
                    B[j, i] = R[j] * (self.x[j]) ** i

            # ff' relations
            for i in range(self.ffprime_order):
                for j in range(N):
                    B[j, i + self.pprime_order] = (1 / (mu_0 * R[j])) * (self.x[j]) ** i

            x_z = np.gradient(np.reshape(self.x, (self.nx, self.ny), order='F'), dR, dZ)[1].flatten(order='F')
            for j in range(N):
                psum = 0
                ffsum = 0
                for i in range(self.pprime_order):
                    if i == 0:
                        pass
                    else:
                        psum += self.c[i] * math.comb(i, 1) * self.x[j] ** (i - 1)

                for i in range(self.ffprime_order):
                    if i == 0:
                        pass
                    else:
                        ffsum += self.c[i + self.pprime_order] * math.comb(i, 1) * self.x[
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
        """
        Find the matrix to perform least squares on

        Parameters
        ----------
        A - Plasma respoonse Matrix

        Returns
        -------
        E - operator matrix
        """
        B = np.identity(self.Gcoil.shape[1])
        C = np.zeros((self.Gcoil.shape[1], A.shape[1]))
        if self.VesselCurrents:
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
        jtor - 2d current density matrix

        Returns
        -------
        x - normalised psi
        mask - mask generated form xpoints
        """

        self.check_limited = True
        self._updateBoundaryPsi()

        if jtor is not None:
            from scipy.integrate import trapz
            self.Jtor = jtor  # update eq jtor attirbute

            # Using Elliptical Solver to calculate plasma_psi from plasma jtor profile
            rhs = -mu_0 * self.R * jtor

            # dont use plasma psi, calculate expected from new jtor and greens
            R = self.R
            Z = self.Z
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
                self.plasma_psi[x, y] = rhs[x, y]

            plasma_psi = self._solver(self.plasma_psi, rhs)
            self._updatePlasmaPsi(plasma_psi)

        # Fetch total psi (plasma + coils)
        psi = self.psi()

        # Calculating Locations for axis and boundaries
        opt, xpt = critical.find_critical(self.R, self.Z, psi)
        psi_axis = opt[0][2]

        if xpt:
            psi_bndry = xpt[0][2]
            mask = critical.core_mask(self.R, self.Z, psi, opt, xpt)
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
        m - number of radial finite elements
        n - number of vertical finite elements
        Returns
        -------
        T - finite element basis matrix for current initialisation
        """
        m += 2
        n += 2
        R = self.R.flatten(order='F')
        Z = self.Z.flatten(order='F')
        a, b, c, d = self.Rmin, self.Rmax, self.Zmin, self.Zmax
        dR = (b - a) / (m - 1)
        dZ = (d - c) / (n - 1)
        r_FE = np.linspace(a, b, m)
        z_FE = np.linspace(c, d, n)

        T = np.zeros((self.nx * self.ny, (m - 2) * (n - 2)))
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

    # Reconstruction Algorithm
    def reconstruct(self, pause=0.0001):
        """
        Parameters
        ----------
        tokamak - tokamak object
        M - measurements
        sigma - measurement uncertainties
        pprime_order - number of polynomial coefficients for pprime model
        ffprime_order - number of polynomial coefficients for ffprime model
        tolerance - value beneath which the chi squared value will stop the iteration
        Gplasma - Plasma Response Greens matrix
        Gcoil - Coil Greens matrix
        Gvessel - Eigenbasis response Greens matrix
        show - option to display psi through the iterations
        pause - time to wait between iterations
        VesselCurrents - Used when tokamak has vessel filaments. Allows calculation of induced currents in tokamak wall
        VerticalControl - option for use of the vertical control found in ferron's paper
        CurrentInitialisation - option for using finite element method to optimise initial guess for jtor
        FittingWeight - option to apply a fitting weight vector to scale lstsq calculation
        """

        # start iterative loop
        self.it = 0  # iteration number
        convergence = False
        while not convergence:

            # Initialisation
            if self.it == 0:
                jtor_2d = None

                if self.show:
                    import matplotlib.pyplot as plt
                    from .plotting import plotEquilibrium

                    if pause > 0.0:
                        # No axis specified, so create a new figure
                        fig, axs = plt.subplots(1,2, sharex=True, sharey=True)

                if self.VesselCurrents:
                    self.Gvessel = self.Gfil @ self.tokamak.eigenbasis

                # Performs plasma current density initialisation
                if self.CurrentInitialisation:
                    jtor_1d = self.initialise_plasma_current()
                    jtor_2d = np.reshape(jtor_1d, (self.nx, self.ny),order='F')

                if self.FittingWeight:
                    F = self.get_F()

            else:

                if self.show:
                    # Plot state of plasma equilibrium
                    if pause < 0:
                        fig = plt.figure()
                        axis = fig.add_subplot(111)
                    else:
                        axs[0].clear()
                        axs[1].clear()

                    plotEquilibrium(self, axis=axs[0], show=False)
                    if self.Jtor is not None:
                        axs[1].imshow(np.rot90(jtor_2d), extent=(0.1,2,-1,1), cmap=plt.get_cmap('jet'))

                    if pause < 0:
                        # Wait for user to close the window
                        plt.show()
                    else:
                        # Update the canvas and pause
                        # Note, a short pause is needed to force drawing update
                        axs[0].figure.canvas.draw()
                        axs[1].figure.canvas.draw()
                        plt.pause(pause)

                # Use B & C to calculate Jtor matrix
                #assumes vertical control is on
                if self.is_limited:
                    B = self.apply_mask(B, mask)
                if self.VesselCurrents:
                    jtor_1d = B @ self.c[:-(len(self.tokamak.coils) + self.Gvessel.shape[1])]
                else:
                    jtor_1d = B @ self.c[:-(len(self.tokamak.coils))]

                jtor_2d = np.reshape(jtor_1d, (self.nx, self.ny),order='F')

            # Calculate Psi values with elliptical solver
            x_2d, mask = self.get_x(jtor_2d)
            self.x = x_2d.flatten('F')

            # Calculate B and apply mask
            B = self.get_B()
            if not self.is_limited:
                B = self.apply_mask(B, mask)

            # Calculating operator matrix A
            A = self.Gplasma @ B
            E = self.get_E(A)

            # Performing least squares calculation for c
            if self.FittingWeight:
                self.c = scipy.linalg.lstsq(F @ E, F @ self.M)[0]
            else:
                self.c = scipy.linalg.lstsq(E, self.M)[0]

            # Updating the Coil Currents and Vessel Currents
            self.update_currents()

            # Take Diagnostics and Perform Convergence Test
            self.H = E @ self.c

            """Not sure if this should be included or not"""

            # if self.VerticalControl and self.it > 0:
            #    for i, val in enumerate(self.H):
            #        self.H[i] += E[i, self.pprime_order + self.ffprime_order] * self.c[self.pprime_order + self.ffprime_order] * (-1)

            """ UP TO HERE"""


            self.check_incorrect_sensors()

            convergence = self.convergence_test(self.tolerance)
            self.it += 1

    # Methods for checking performance
    def check_incorrect_sensors(self):
        if self.it>1:
            self.tokamak.takeMeasurements(self)
            sensors_measure = []
            for i in range(self.Gplasma.shape[0]):
                sensors_measure.append(self.tokamak.sensors[i].measurement)

            for i in range(self.Gplasma.shape[0]):
                plasma_response = np.dot(self.Gplasma[i], self.Jtor.flatten('F'))
                if self.VesselCurrents:
                    coil_response = np.dot(self.Gcoil[i], self.c[-(
                                self.Gvessel.shape[1] + self.Gcoil.shape[1]):-
                                                                 self.Gvessel.shape[
                                                                     1]])
                    vessel_response = np.dot(self.Gvessel[i],
                                             self.c[-self.Gvessel.shape[1]:])
                    if not np.isclose(
                            plasma_response + coil_response + vessel_response,
                            sensors_measure[i], rtol=0.015):
                        print(self.tokamak.sensors[i].name, 'Incorrect', self.H[i],
                              plasma_response + coil_response + vessel_response,
                              sensors_measure[i])
                        if self.test:
                            assert 1==2

                else:
                    coil_response = np.dot(self.Gcoil[i],
                                           self.c[-(self.Gcoil.shape[1]):])

                    if not np.isclose(plasma_response + coil_response,
                                      sensors_measure[i], rtol=0.01):
                        print(self.tokamak.sensors[i].name, 'Incorrect',plasma_response + coil_response,sensors_measure[i])
                        if self.test:
                            assert 1==2

    def plot_filaments(self):
        plt.figure()

        for filament in self.tokamak.vessel:
            for subfil in filament.filaments:
                if subfil.current >= 0:
                    plt.scatter(subfil.R, subfil.Z, color='red',
                                s=abs(subfil.current))
                else:
                    plt.scatter(subfil.R, subfil.Z, color='black',
                                s=abs(subfil.current))

        plt.show()

    def print_reconstruction(self):
        """
        Print data from reconstruction
        """

        print('Completed in ', self.it, 'iterations, with Chi Squared =', self.chi)

        if self.VesselCurrents:
            self.dz = self.c[-len(self.tokamak.coils) - 1 - self.Gvessel.shape[1]]
        else:
            self.dz = self.c[-len(self.tokamak.coils) - 1]

        print('Vertical Control Parameter: dz = ', self.dz)

        print(' ')
        print('Coefficients (alpha_i, dz, coil currents, vessel basis coefficients)')
        print(self.c)

        # Printing final constructed values
        print(' ')
        print('Reconstruction Sensor Diagnostics')
        self.tokamak.printMeasurements(equilibrium=self)
        self.tokamak.printCurrents()

        for sensor, val1, val2 in zip(self.tokamak.sensors, self.M, self.H):
            print(sensor.name, val1, val2, sensor.measurement)

        for (name, coil), val1, val2 in zip(self.tokamak.coils, self.M[-len(self.tokamak.coils):], self.H[-len(self.tokamak.coils):]):
            print(name, val1, val2, coil.current)

    # Methods for solving
    def solve_from_tokamak(self):
        self.take_Measurements_from_tokamak()
        self.reconstruct()
        self.print_reconstruction()
        #self.plot(show=self.show)


    def solve_from_dictionary(self, measurement_dict, sigma_dict):
        self.take_Measurements_from_dictionary(measurement_dict, sigma_dict)
        self.initialise_coil_current()
        self.reconstruct()
        self.print_reconstruction()
        #self.plot(show=self.show)

        self.H=None
        self.M=None
        self.Jtor=None
        self.dz=None
        self.chi=None
        self.sigma=None
        self.x=None
        self.c=None


# Creating an equilibrium
def generate_Measurements(tokamak, alpha_m, alpha_n, Rmin=0.1, Rmax=2, Zmin=-1,
                          Zmax=1, nx=65, ny=65, x_z1=0.6, x_z2=-0.6, x_r1=1.1,
                          x_r2=1.1, show=True):
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
    eq - equilibrium object

    """

    eq = Equilibrium(tokamak=tokamak,
                                 Rmin=Rmin, Rmax=Rmax,  # Radial domain
                                 Zmin=Zmin, Zmax=Zmax,  # Height range
                                 nx=nx, ny=ny,  # Number of grid points
                                 boundary=boundary.freeBoundary)  # Boundary condition

    profiles = jtor.ConstrainPaxisIp(eq, 1e3,
                                     2e5,
                                     2,
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


