"""
Classes and routines to reconstruct magnetic profiles from sensor data

License
-------

Copyright 2022 Angus Gibby,  Email: angus.gibby@new.ox.ac.uk

This file is part of FreeGS.

FreeGS is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FreeGS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGS.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import scipy
import math
from . import critical, plotting, jtor, control, picard, boundary, filament_coil, machine
from .equilibrium import Equilibrium
from .machine import Coil, ShapedCoil, FilamentCoil, RogowskiSensor, PoloidalFieldSensor, FluxLoopSensor, Filament, Passive, Wall, Circuit, Machine
from .gradshafranov import Greens, GreensBr, GreensBz
from shapely.geometry import Point, Polygon, LineString, LinearRing
from scipy.special import ellipk, ellipe
from scipy.constants import mu_0
import matplotlib.pyplot as plt
import pickle

class Reconstruction(Equilibrium):
    def __init__(self, pprime_order, ffprime_order, tolerance=1e-8, coil_weight=50, use_VesselCurrents=True, use_VerticalControl=True, use_CurrentInitialisation=True, show=True, test=False, **kwargs):
        Equilibrium.__init__(self, **kwargs)
        self.test = test

        # Defining Reconstruction Parameters
        self.pprime_order = pprime_order
        self.ffprime_order = ffprime_order
        self.use_VesselCurrents = use_VesselCurrents
        self.use_VerticalControl = use_VerticalControl
        self.use_CurrentInitialisation = use_CurrentInitialisation
        self.tolerance = tolerance
        self.coil_weight = coil_weight
        self.show=show

        self.dR = self.R[1, 0] - self.R[0, 0]
        self.dZ = self.Z[0, 1] - self.Z[0, 0]

        # Creating Reconstruction Grid and Greens functions
        self.generate_Greens()

        # Reconstruction attributes
        self.psi_norm = None # normalised psi
        self.c = None # coefficients
        self.check_limited = True
        self.pause = 0.0001

    # Methods for building measurment and uncertainty matrices
    def take_Measurements_from_dictionary(self, measurement_dict, sigma_dict):
        """
        Function called to generate measurement matrix and sigma
        Also give new tokamak object the coil currents

        Parameters
        ----------
        tokamak - tokamak object
        measurement_dict - dictionary containing measurements
        sigma_dict - dictionry containing uncertainty values

        Returns
        -------
        M - Returns measurement matrix
        sigma - Returns uncertainty values
        """

        M = []
        sigma = []

        # Giving sensor measurements to M
        for sensor in self.tokamak.sensors:
            M.append([measurement_dict[sensor.name]])
            sigma.append([sigma_dict[sensor.name]])

        # Giving coil currents to M
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
        tokamak - tokamak object, contained within are sensors and coils
                  which are preloaded with the measurement values for reconstruction

        Returns
        -------
        M - Returns measurement matrix
        sigma - Returns uncertainty values
        """

        M = []
        sigma = []

        # Adding sensor measurements to M
        for sensor in self.tokamak.sensors:
            M.append([sensor.measurement])
            sigma.append([1 / sensor.weight])

        # Adding Coil Currents to M
        for name, coil in self.tokamak.coils:
            M.append([coil.current])
            sigma.append([1/self.coil_weight])

        self.M = M
        self.sigma = sigma

    # Pass the coil current sto the equilibrium object
    def initialise_coil_current(self):
        """
        Method for giving the tokamak object the correct currents for reconstruction
        """
        coil_currents = self.M[-self.tokamak.n_coils:]
        for index, (name, coil) in enumerate(self.tokamak.coils):
            coil.current = coil_currents[index][0]

    # Calculate the Current initialisation T matrix
    def initialise_plasma_current(self, m=5, n=5):
        """
        Function for calculating the initial jtor

        Parameters
        ----------
        Gplasma - greens plasma response matrix
        M_plasma - plasma contribution to measurements
        FEmatrix - Finite Element basis matrix

        Returns
        -------
        jtor - current density matrix
        """

        #Finding plasma dependence of measurements
        M_plasma = self.get_M_plasma()

        # Creates finite element grid
        FEmatrix = self.get_FiniteElements(m=m,n=n)

        #Uses least squares solver to find coefficients and subsequently jtor
        coefs = scipy.linalg.lstsq(self.Gplasma @ FEmatrix, M_plasma)[0]
        jtor = FEmatrix @ coefs
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

        # Finds the coil contirbution to the measurements
        self.tokamak.takeMeasurements()

        # Finds the plasma (+vessel) contribution to measurements
        for i, sensor in enumerate(self.tokamak.sensors):
            M_plasma.append(self.M[i][0] - sensor.measurement)

        return M_plasma

    # Applying mask to a 2d matrix
    def apply_mask(self, Matrix, mask):
        if isinstance(mask, np.ndarray):
            for index in range(Matrix.shape[1]):
                Matrix[:,index] *= mask.flatten(order='F')
        return Matrix

    # Updating vessel and filament currents
    def update_currents(self):

        # Loops through coils, updating the current with the new calculated values
        coil_currents = self.get_index(self.c, 'coils')
        for i, (name, circuit) in enumerate(self.tokamak.coils):
            circuit.current = coil_currents[i][0]

        # Updates filament currents if they are being used
        if self.use_VesselCurrents:
            fil_currents = self.tokamak.eigenbasis @ self.get_index(self.c, 'vessel')
            self.tokamak.updateVesselCurrents(fil_currents)


    #Perform a Chi Squared test
    def convergence_test(self):
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

        # Chi Squared Test
        new_chi = 0
        for i in range(len(self.M)):
            new_chi += ((self.M[i] - self.H[i]) / self.sigma[i]) ** 2

        print('Chi Squared =', new_chi)

        # Convergence Conditions
        convergence = False

        # Convergence must occur after at least 2 iterations, and the chi squared must be same as previous up to 4 sig fig
        if self.it>2 and (new_chi <= self.tolerance or float('%.3g' % new_chi) == float('%.3g' % self.chi)):
            convergence = True

        # Break due to runtime
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
        PlasmaGreens = np.zeros((self.tokamak.n_sensors, self.nx*self.ny))

        for w, sensor in enumerate(self.tokamak.sensors):
            greens_matrix = sensor.get_PlasmaGreensRow(self)
            greens_row = greens_matrix.flatten(order='F')
            PlasmaGreens[w] = greens_row
        self.Gplasma = PlasmaGreens

    def get_CoilGreens(self):
        """
        Function for calculating greens matrix for the coil contribution to sensor measurements.
        Runs on initialisation only

        Parameters
        ------
        tokamak - machine with location of coils


        Returns
        -------
        Gcoil - Coil response matrix

        """
        CoilGreens = np.zeros((self.tokamak.n_sensors, self.tokamak.n_coils))


        for w, sensor in enumerate(self.tokamak.sensors):
            for coil_num, (coil_name, coil) in enumerate(self.tokamak.coils):
                CoilGreens[w, coil_num] = sensor.get_CoilGreensRow(coil)

        self.Gcoil = CoilGreens

    def get_VesselGreens(self):
        """
        Function for calculating greens matrix for the vessel current contribution to sensor measurements.
        Runs on initialisation only

        Parameters
        ------
        tokamak - machine with location of filaments


        Returns
        -------
        Gfil - filament response matrix

        """

        FilamentGreens = np.zeros((self.tokamak.n_sensors, self.tokamak.n_fils))

        for w, sensor in enumerate(self.tokamak.sensors):
            fil_index = 0
            for passive in self.tokamak.vessel:
                for fil in passive.filaments:
                        FilamentGreens[w, fil_index] = sensor.get_VesselGreensRow(fil)
                        fil_index += 1

        self.Gvessel = FilamentGreens @ self.tokamak.eigenbasis

    def generate_Greens(self):
        """
        Method for calling the greens function methods
        """
        self.get_PlasmaGreens()
        self.get_CoilGreens()
        if self.use_VesselCurrents:
            self.get_VesselGreens()

    # Basis Matrix B (N x nc)
    def get_BasisMatrix(self):
        """
        Function for calculating the basis matrix, runs every iteration

        Parameters
        ----------
        psi_norm - normalised psi
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

        if self.use_VerticalControl and self.it>0:
            B = np.zeros((N, nc + 1))
            # p' Coefficients
            for i in range(self.pprime_order):
                B[:, i] = R * self.psi_norm ** i

            # ff' relations
            for i in range(self.ffprime_order):
                B[:, i + self.pprime_order] = (1 / (
                            mu_0 * R)) * self.psi_norm ** i


            # This next section corresponds to the vertical contorl
            # The derivative of psi_norm wrt to Z is taken
            # Then as per Ferron, the linear dependence of the modified free functions is taken
            x_z = np.gradient(np.reshape(self.psi_norm, (self.nx, self.ny), order='F'), self.dR, self.dZ)[1].flatten(order='F')
            psum = 0
            ffsum = 0

            for i in range(1,self.pprime_order):
                psum += self.c[i] * math.comb(i, 1) * self.psi_norm ** (i - 1)

            for i in range(1,self.ffprime_order):
                ffsum += self.c[i + self.pprime_order] * math.comb(i, 1) * self.psi_norm ** (i - 1)

                B[:, nc] = x_z * (R * psum + 1 / (mu_0 * R) * ffsum)

        else:
            B = np.zeros((N, nc))
            # p' Coefficients
            for i in range(self.pprime_order):
                    B[:, i] = R * self.psi_norm ** i

            # ff' relations
            for i in range(self.ffprime_order):
                    B[:, i + self.pprime_order] = (1 / (mu_0 * R)) * self.psi_norm ** i

        return B

    # Finding total operator matrix
    def get_SystemMatrix(self, A):
        """
        Find the matrix E upon which Ec = M (the one characterising the system)

        Parameters
        ----------
        A - Plasma response Matrix

        Returns
        -------
        E - operator matrix
        """
        B = np.identity(self.Gcoil.shape[1])
        C = np.zeros((self.Gcoil.shape[1], A.shape[1]))
        if self.use_VesselCurrents:
            D = np.zeros((self.Gcoil.shape[1], self.Gvessel.shape[1]))
            E = np.block([[A, self.Gcoil, self.Gvessel], [C, B, D]])
        else:
            E = np.block([[A, self.Gcoil], [C, B]])
        return E

    # Calculate a 2 dimensional normalised psi
    def get_psi_norm(self):
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

        if self.Jtor is not None:
            self.solve(Jtor=self.Jtor)

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
    def get_fittingWeightVector(self):
        """
        Uses the inputted sigma vector to create the diagonal fitted weight matrix
        Weighting = 1 / sigma

        Parameters
        ----------
        sigma - vector containing measurement uncertainties

        """
        return np.diag([val[0]**(-1) for val in self.sigma])

    # Calculate T
    def get_FiniteElements(self, m, n):
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

    # Indexing function
    def get_index(self, Vector, indexing):

        # Finding the corresponding index for the string
        if indexing == 'plasma':
            if self.use_VerticalControl and self.it>1:
                return Vector[:(self.pprime_order+self.ffprime_order+1)]
            else:
                return Vector[:(self.pprime_order+self.ffprime_order)]
        if indexing == 'coils':
            if self.use_VesselCurrents:
                return Vector[-(self.tokamak.n_coils+self.tokamak.n_basis):-self.tokamak.n_basis]
            else:
                return Vector[-self.tokamak.n_coils:]

        if indexing == 'vessel':
            return Vector[-self.tokamak.n_basis:]

    # Reconstruction Algorithm
    def reconstruct(self):
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
        use_VesselCurrents - Used when tokamak has vessel filaments. Allows calculation of induced currents in tokamak wall
        use_VerticalControl - option for use of the vertical control found in ferron's paper
        use_CurrentInitialisation - option for using finite element method to optimise initial guess for jtor
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

                    if self.pause > 0.0:
                        # No axis specified, so create a new figure
                        fig, axs = plt.subplots(1,2, sharex=True, sharey=True)

                # Performs plasma current density initialisation
                if self.use_CurrentInitialisation:
                    jtor_1d = self.initialise_plasma_current()
                    self.Jtor = np.reshape(jtor_1d, (self.nx, self.ny),order='F')

                F = self.get_fittingWeightVector()

            else:

                if self.show:
                    self.plot_equilibria(fig, axs)

                # Use B & C to calculate Jtor matrix
                jtor_1d = Basis @ self.get_index(self.c, 'plasma')
                self.Jtor = np.reshape(jtor_1d, (self.nx, self.ny),order='F')

            # Calculate Psi values with elliptical solver
            x_2d, mask = self.get_psi_norm()
            self.psi_norm = x_2d.flatten('F')

            # Calculate B and apply mask
            Basis = self.get_BasisMatrix()
            Basis = self.apply_mask(Basis, mask)

            # Calculating operator matrix A
            A = self.Gplasma @ Basis
            SystemMatrix = self.get_SystemMatrix(A)

            # Performing least squares calculation for c
            self.c = scipy.linalg.lstsq(F @ SystemMatrix, F @ self.M)[0]

            # Updating the Coil Currents and Vessel Currents
            self.update_currents()

            # Take Diagnostics and Perform Convergence Test
            self.H = SystemMatrix @ self.c

            convergence = self.convergence_test()
            self.check_incorrect_sensors()

            self.it += 1

    # Methods for checking performance
    def check_incorrect_sensors(self):

        # Runs through sensors, ensures that the value generated by H is within 2% of what the sensor measures
        if self.it>1:
            self.tokamak.takeMeasurements(self)
            sensors_measure = []
            for i in range(self.tokamak.n_sensors):
                sensors_measure.append(self.tokamak.sensors[i].measurement)

            for i in range(self.tokamak.n_sensors):
                plasma_response = np.dot(self.Gplasma[i], self.Jtor.flatten('F'))
                coil_response = np.dot(self.Gcoil[i], self.get_index(self.c, 'coils'))
                if self.use_VesselCurrents:
                    vessel_response = np.dot(self.Gvessel[i], self.get_index(self.c, 'vessel'))
                else:
                    vessel_response = 0

                if not np.isclose(plasma_response + coil_response + vessel_response,sensors_measure[i], rtol=0.02):
                    print(self.tokamak.sensors[i].name, 'Incorrect', self.H[i],plasma_response + coil_response + vessel_response, sensors_measure[i])
                    if self.test:
                        assert np.isclose(plasma_response + coil_response + vessel_response,sensors_measure[i], rtol=0.02)

    def plot_filaments(self):

        # Plots the eigenbasis for the machine
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

    def plot_equilibria(self, fig, axs):
        # Plot state of plasma equilibrium
        if self.pause < 0:
            fig = plt.figure()
            axis = fig.add_subplot(111)
        else:
            axs[0].clear()
            axs[1].clear()

        plotting.plotEquilibrium(self, axis=axs[0], show=False)
        if self.Jtor is not None:
            axs[1].imshow(np.rot90(self.Jtor), extent=(0.1, 2, -1, 1),
                          cmap=plt.get_cmap('jet'))

        if self.pause < 0:
            # Wait for user to close the window
            plt.show()
        else:
            # Update the canvas and pause
            # Note, a short pause is needed to force drawing update
            axs[0].figure.canvas.draw()
            axs[1].figure.canvas.draw()
            plt.pause(self.pause)

    def print_reconstruction(self):
        """
        Print data from reconstruction
        """

        print('Completed in ', self.it, 'iterations, with Chi Squared =', self.chi)

        if self.use_VerticalControl:
            self.dz = self.get_index(self.c, 'plasma')[-1]

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

        for (name, coil), val1, val2 in zip(self.tokamak.coils, self.M[-self.tokamak.n_coils:], self.H[-self.tokamak.n_coils:]):
            print(name, val1, val2, coil.current)

    # Methods for solving
    def solve_from_tokamak(self):
        self.take_Measurements_from_tokamak()
        self.reconstruct()
        self.print_reconstruction()
        #self.plot(show=self.show)

        if self.test:
            self.H=None
            self.M=None
            self.Jtor=None
            self.dz=None
            self.chi=None
            self.sigma=None
            self.psi_norm=None
            self.c=None
            self.it=0
            self.is_limited = False
            self.psi_norm = None
            self.psi_axis = None
            self.psi_bndry = None
            self.psi_limit = None

    def solve_from_dictionary(self, measurement_dict, sigma_dict):
        self.take_Measurements_from_dictionary(measurement_dict, sigma_dict)
        self.initialise_coil_current()
        self.reconstruct()
        self.print_reconstruction()
        #self.plot(show=self.show)

        if self.test:
            self.H=None
            self.M=None
            self.Jtor=None
            self.dz=None
            self.chi=None
            self.sigma=None
            self.psi_norm=None
            self.c=None
            self.it=0
            self.is_limited = False
            self.psi_norm = None
            self.psi_axis = None
            self.psi_bndry = None
            self.psi_limit = None


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


