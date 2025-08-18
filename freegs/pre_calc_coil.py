"""
Classes and routines to represent coils and circuits

License
-------

Copyright 2022 Chris Marsden, Tokamak Energy. Email: chris.marsden@tokamakenergy.co.uk

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
from scipy.interpolate import RectBivariateSpline

from . import polygons
from .coil import Coil


class PreCalcCoil(Coil):
    """
    This class represents a coil whose Green's functions have
    already been calculated by some external code. This is useful
    in modelling coils whose internal structure may be complex and
    whose current distribution may be highly non-uniform.

    The user needs to supply information on the R,Z grids that the
    Green's functions have been pre-calculated on, as well as the
    Br, Bz and psi data on said R,Z grid.

    public members
    --------------

    R, Z    - Location of the coil
    current - current in the coil in Amps
    turns   - Number of turns
    control - enable or disable control system
    area    - Cross-section area in m^2

    The total toroidal current carried by the coil is current * turns
    """

    # A dtype for converting to Numpy array and storing in HDF5 files
    dtype = np.dtype(
        [
            ("RZlen", int),  # Length of the R and Z arrays
            ("R", "10f8"),  # Note: Up to 10 points
            ("Z", "10f8"),  # Note: Up to 10 points
            ("current", np.float64),
            ("turns", int),
            ("control", bool),
        ]
    )

    def __init__(
        self,
        shape,
        Rgrid,
        Zgrid,
        mapBr,
        mapBz,
        mapPsi,
        current=0.0,
        turns=1,
        control=True,
    ):
        """
        Inputs
        ------
        shape  - outline of the coil shape as a list of points [(r1,z1), (r2,z2), ...]
                 Must have more than two points

        Rgrid   - 1D array of R coords that maps are calculated on.
        Zgrid   - 1D array of Z coords that maps are calculated on.
        mapBr   - 1D array of Br calculated on Rgrid,Zgrid for the coil @ 1A-turn.
        mapBz   - 1D array of Bz calculated on Rgrid,Zgrid for the coil @ 1A-turn.
        mapPsi  - 1D array of Psi calculated on Rgrid,Zgrid for the coil @ 1A-turn.
        current - The current in the circuit. The total current is current * turns.
        turns   - Number of turns in point coil(s) block. Total block current is current * turns.
        control - Enable or disable control system.

        """
        assert len(shape) > 2

        # Find the geometric middle of the coil
        # The R,Z properties have accessor functions to handle modifications
        self._R_centre = sum(r for r, z in shape) / len(shape)
        self._Z_centre = sum(z for r, z in shape) / len(shape)

        self.current = current
        self.turns = turns
        self.control = control
        self._area = abs(polygons.area(shape))
        self.shape = shape
        self._points = np.array([(r, z) for r, z in self.shape])

        # Data for the pre-calculated Green's functions
        self.Rgrid = np.transpose(Rgrid)[:, 0]
        self.Zgrid = np.transpose(Zgrid)[0, :]
        self.mapPsi = np.transpose(np.asarray(mapPsi))
        self.mapBr = np.transpose(np.asarray(mapBr))
        self.mapBz = np.transpose(np.asarray(mapBz))

        # Interpolators for the pre-calculated Green's functions
        self.cPsi = RectBivariateSpline(self.Rgrid, self.Zgrid, self.mapPsi)
        self.cBr = RectBivariateSpline(self.Rgrid, self.Zgrid, self.mapBr)
        self.cBz = RectBivariateSpline(self.Rgrid, self.Zgrid, self.mapBz)

    def controlPsi(self, R, Z):
        """
        Calculate poloidal flux at (R,Z) due to a unit current.
        """

        if isinstance(R, float | int):
            result = self.cPsi(R, Z)[0][0]
        else:
            result = self.cPsi(R, Z, grid=False)

        return result

    def controlBr(self, R, Z):
        """
        Calculate radial magnetic field Br at (R,Z) due to a unit current.
        """

        if isinstance(R, float | int):
            result = self.cBr(R, Z)[0][0]
        else:
            result = self.cBr(R, Z, grid=False)

        return result

    def controlBz(self, R, Z):
        """
        Calculate vertical magnetic field Br at (R,Z) due to a unit current.
        """

        if isinstance(R, float | int):
            result = self.cBz(R, Z)[0][0]
        else:
            result = self.cBz(R, Z, grid=False)

        return result

    def __repr__(self):
        return f"PreCalcCoil({self.shape}, current={self.current:.1f}, turns={self.turns}, control={self.control})"

    @property
    def R(self):
        """
        Major radius of the coil in m
        """
        return self._R_centre

    @R.setter
    def R(self, Rnew):
        # Need to shift all points
        Rshift = Rnew - self._R_centre
        self._points = [(r + Rshift, z) for r, z in self._points]
        self._R_centre = Rnew

    @property
    def Z(self):
        """
        Height of the coil in m
        """
        return self._Z_centre

    @Z.setter
    def Z(self, Znew):
        # Need to shift all points
        Zshift = Znew - self._Z_centre
        self._points = [(r, z + Zshift) for r, z in self._points]
        self._Z_centre = Znew

    @property
    def area(self):
        return self._area

    @area.setter
    def area(self, area):
        raise ValueError("Area of a PreCalcCoil is fixed")

    def plot(self, axis=None, show=False):
        """
        Plot the coil shape, using axis if given
        """
        import matplotlib.pyplot as plt

        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)

        r = [r for r, z in self.shape]
        z = [z for r, z in self.shape]
        axis.fill(r, z, color="gray")
        axis.plot(r, z, color="black")

        # Quadrature points
        # rquad = [r for r,z,w in self._points]
        # zquad = [z for r,z,w in self._points]
        # axis.plot(rquad, zquad, 'ro')

        return axis

    def to_numpy_array(self):
        """
        Helper method for writing output
        """
        RZlen = len(self.shape)
        R = np.zeros(10)
        Z = np.zeros(10)
        R[:RZlen] = [R for R, Z in self.shape]
        Z[:RZlen] = [Z for R, Z in self.shape]

        return np.array(
            (RZlen, R, Z, self.current, self.turns, self.control),
            dtype=self.dtype,
        )

    def __ne__(self, other):
        return not self == other
