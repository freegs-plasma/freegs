"""
Define a class of coil which contains a uniform current density
over a shaped region.

License
-------

Copyright 2019 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk

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

from . import quadrature
from . import polygons
from .gradshafranov import Greens, GreensBr, GreensBz

import numpy as np

from .coil import Coil


class ShapedCoil(Coil):
    """
    Represents a coil with a specified shape

    public members
    --------------

    R, Z - Location of the point coil/Locations of coil filaments
    current - current in the coil(s) in Amps
    turns   - Number of turns if using point coils
    control - enable or disable control system
    area    - Cross-section area in m^2

    The total toroidal current carried by the coil block is current * turns
    """

    # A dtype for converting to Numpy array and storing in HDF5 files
    dtype = np.dtype(
        [
            (str("R"), np.float64),
            (str("Z"), np.float64),
            (str("current"), np.float64),
            (str("turns"), np.int),
            (str("control"), np.bool),
            (str("mirror"), np.bool),
        ]
    )

    def __init__(self, shape, current=0.0, turns=1, control=True, npoints=6):
        """
        Inputs
        ------
        shape     outline of the coil shape as a list of points [(r1,z1), (r2,z2), ...]
                  Must have more than two points
        current   The current in the circuit. The total current is current * turns
        turns     Number of turns in point coil(s) block. Total block current is current * turns
        control   enable or disable control system
        npoints   Number of quadrature points per triangle. Valid choices: 1, 3, 6

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

        # The quadrature points to be used
        self._points = quadrature.polygon_quad(shape, n=npoints)

    def controlPsi(self, R, Z):
        """
        Calculate poloidal flux at (R,Z) due to a unit current
        """
        result = 0.0
        for R_fil, Z_fil, weight in self._points:
            result += Greens(R_fil, Z_fil, R, Z) * weight
        return result

    def controlBr(self, R, Z):
        """
        Calculate radial magnetic field Br at (R,Z) due to a unit current
        """
        result = 0.0
        for R_fil, Z_fil, weight in self._points:
            result += GreensBr(R_fil, Z_fil, R, Z) * weight
        return result

    def controlBz(self, R, Z):
        """
        Calculate vertical magnetic field Bz at (R,Z) due to a unit current
        """
        result = 0.0
        for R_fil, Z_fil, weight in self._points:
            result += GreensBz(R_fil, Z_fil, R, Z) * weight
        return result

    def __repr__(self):
        return "ShapedCoil({0}, current={1:.1f}, turns={2}, control={3})".format(
            self.shape, self.current, self.turns, self.control
        )

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
        self._points = [(r + Rshift, z, w) for r, z, w in self._points]
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
        self._points = [(r, z + Zshift, w) for r, z, w in self._points]
        self._Z_centre = Znew

    @property
    def area(self):
        return self._area

    @area.setter
    def area(self, area):
        raise ValueError("Area of a ShapedCoil is fixed")

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
        axis.fill(r, z, color="blue")

        # Quadrature points
        # rquad = [r for r,z,w in self._points]
        # zquad = [z for r,z,w in self._points]
        # axis.plot(rquad, zquad, 'ro')

        return axis
