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
from . import polygons
from shapely import geometry
from .coil import Coil, AreaCurrentLimit
from .gradshafranov import Greens, GreensBr, GreensBz


def populate_with_fils(shape, Nfils):
    """
    Takes the 2D cross section of a PF coil, via shape, and populates it
    (somewhat) uniformally with ~Nfils point sources representing filaments.
    """

    Rshape = np.array([i[0] for i in shape])
    Zshape = np.array([i[1] for i in shape])
    Rshape = np.append(Rshape, Rshape[0])
    Zshape = np.append(Zshape, Zshape[0])
    rmin = np.min(Rshape)
    rmax = np.max(Rshape)
    zmin = np.min(Zshape)
    zmax = np.max(Zshape)

    my_polygon = geometry.Polygon(shape)

    """
    How this works: Start by roughly estimating the d = dR, dZ spacing
    between the points, using the shape area and Nfils.
    """

    A = abs(polygons.area(shape))
    d = np.sqrt(A / Nfils)

    """
    Now we create a (staggered) grid across the bounding box containing
    the supplied shape. We then scan through each point on the grid and
    check if it lies inside the shape. We count the number of points that
    lie inside the shape. If this is less than Nfils, increase the grid
    resolution a little until eventually the number of points inside the grid
    => Nfils. Once this is obtained, linearly interpolate in dR,dZ to better
    estimate the d(=dR,dZ) that will give as close to Nfils points inside the shape
    as possible.
    """

    # Track d vs nInside
    d_data = []
    nInside_data = []

    nInside = 0

    while nInside < Nfils:

        nInside = 0
        i = 0  # Row counter used for staggering every other row
        # Create the (non-staggered) grid points for the given d(=dR,dZ)
        Rgrid = np.arange(rmin, rmax, d, dtype=float)
        Zgrid = np.arange(zmin, zmax, d, dtype=float)

        nR = len(Rgrid)
        nZ = len(Zgrid)

        for i in range(nZ):

            zpoint = Zgrid[i]

            if i % 2:  # Every other row, stagger R coords by 0.5*dR
                offset = 0.5 * d
            else:

                offset = 0.0

            for j in range(nR):

                rpoint = Rgrid[j] + offset

                point = geometry.Point(rpoint, zpoint)
                if my_polygon.contains(point):
                    nInside += 1

        # End of looping over points. Note d and nInside
        d_data.append(d)
        nInside_data.append(nInside)

        # Shrink spacing by 0.5%
        d *= 0.995

    # Finished and have got nInside >= Nfils
    d_data = np.asarray(d_data)
    nInside = np.asarray(nInside_data)

    # Now linearly interpolate to get the d that corresponds
    # to the closest nInside to Nfils
    d_opt = np.interp([Nfils * 1.1], nInside_data, d_data)

    # Now get the locations of the filaments for this d = d_opt where nInside ~ Nfils
    rfils = []
    zfils = []

    # Number of filaments landing inside
    nInside = 0

    i = 0  # Row counter used for staggering every other row
    # Create the (non-staggered) grid points for the given d(=dR,dZ)
    Rgrid = np.arange(rmin, rmax, d_opt, dtype=float)
    Zgrid = np.arange(zmin, zmax, d_opt, dtype=float)

    nR = len(Rgrid)
    nZ = len(Zgrid)

    for i in range(nZ):

        zpoint = Zgrid[i]

        if i % 2:  # Every other row, stagger R coords by 0.5*dR
            offset = 0.5 * d_opt
        else:

            offset = 0.0

        for j in range(nR):

            rpoint = Rgrid[j] + offset

            point = geometry.Point(rpoint, zpoint)

            if my_polygon.contains(point):
                nInside += 1
                rfils.append(rpoint)
                zfils.append(zpoint)

    rfils = np.asarray(rfils)
    zfils = np.asarray(zfils)

    return rfils, zfils


class FilamentCoil(Coil):
    """
    This class represents a coil broken down into multiple
    filaments, with each filament acting as a point source
    of current.

    Note: Filaments are wired in parallel, so a the current
    through a single turn is shared between the filaments.

    public members
    --------------

    R, Z    - Location of the coil
    current - current in each turn of the coil in Amps
    turns   - Number of turns
    control - enable or disable control system
    area    - Cross-section area in m^2

    The total toroidal current carried by the coil is current * turns
    """

    # A dtype for converting to Numpy array and storing in HDF5 files
    dtype = np.dtype(
        [
            (str("RZlen"), int),  # Length of R and Z arrays
            (str("R"), "500f8"),  # Up to 100 points
            (str("Z"), "500f8"),
            (str("current"), np.float64),
            (str("turns"), int),
            (str("control"), bool),
            (str("npoints"), int),
        ]
    )

    def __init__(
        self,
        Rfil=None,
        Zfil=None,
        shape=None,
        current=0.0,
        Nfils=50,
        turns=1,
        control=True,
    ):
        """
        Inputs
        ------
        shape   - Outline of the coil shape as a list of points [(r1,z1), (r2,z2), ...]
                  Must have more than two points. If not provided, plotting the coil
        may not be entirely accurate.

        Rfil, Zfil - Locations of coil filaments (lists/arrays). This is optional.
        If these are not provided then the filaments themselves are populated
        automatically across the cross-section of the coil.

        current - current in each turn of the coil in Amps
        Nfils   - Number of filaments. Only used when Rfil is None.
        turns   - Number of turns in point coil(s) block. Total block current is current * turns
        control - enable or disable control system.

        Note: The number of filaments does not equal the number of turns; each turn
        of the coil might consist of multiple filaments.

        """

        if shape is None:
            # Note: NumPy min/max works with float where builtin doesn't
            minR = np.min(Rfil)
            maxR = np.max(Rfil)
            minZ = np.min(Zfil)
            maxZ = np.max(Zfil)

            shape = [
                (minR, minZ),
                (minR, maxZ),
                (maxR, maxZ),
                (maxR, minZ),
                (minR, minZ),
            ]

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

        if Rfil is None:
            # No filaments provided. Need to populate the coil cross
            # section with -Nfils filaments.
            Rfil, Zfil = populate_with_fils(self.shape, Nfils)

        Rfil = np.asarray(Rfil)
        Zfil = np.asarray(Zfil)
        if Rfil.ndim == 0:
            self.points = [(Rfil, Zfil)]
            self.npoints = 1
        else:
            self.points = np.array(list(zip(Rfil, Zfil)))
            self.npoints = len(Rfil)

    def controlPsi(self, R, Z):
        """
        Calculate poloidal flux at (R,Z) due to a unit current

        Note: This is multiplied by current to get coil Psi
        """
        result = 0.0
        for R_fil, Z_fil in self.points:
            result += Greens(R_fil, Z_fil, R, Z)
        result = result * self.turns / float(self.npoints)
        return result

    def controlBr(self, R, Z):
        """
        Calculate radial magnetic field Br at (R,Z) due to a unit current
        """
        result = 0.0
        for R_fil, Z_fil in self.points:
            result += GreensBr(R_fil, Z_fil, R, Z)
        result = result * self.turns / float(self.npoints)
        return result

    def controlBz(self, R, Z):
        """
        Calculate axial magnetic field Br at (R,Z) due to a unit current
        """
        result = 0.0
        for R_fil, Z_fil in self.points:
            result += GreensBz(R_fil, Z_fil, R, Z)
        result = result * self.turns / float(self.npoints)
        return result

    def __repr__(self):
        return "FilamentCoil({0}, current={1:.1f}, turns={2}, control={3})".format(
            self.points, self.current, self.turns, self.control
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
        self.points = [(r + Rshift, z) for r, z in self.points]
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
        self.points = [(r, z + Zshift) for r, z in self.points]
        self._Z_centre = Znew

    @property
    def area(self):
        return self._area

    @area.setter
    def area(self, area):
        raise ValueError("Area of a FilamentCoil is fixed")

    def plot(self, axis=None, show=False):
        """
        Plot the coil points, using axis if given
        """
        import matplotlib.pyplot as plt

        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)

        r = [r for r, z in self.shape]
        z = [z for r, z in self.shape]
        axis.fill(r, z, color="gray")
        axis.plot(r, z, color="black")

        r = [r for r, z in self.points]
        z = [z for r, z in self.points]
        axis.plot(r, z, "kx")

        # Quadrature points
        # rquad = [r for r,z,w in self._points]
        # zquad = [z for r,z,w in self._points]
        # axis.plot(rquad, zquad, 'ro')

        return axis
