"""
Classes and routines to represent coils and circuits

License
-------

Copyright 2019 Chris Marsden, Tokamak Energy. Email: Chris.Marsden@tokamakenergy.co.uk

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
from .coil import Coil, AreaCurrentLimit
from .gradshafranov import Greens, GreensBr, GreensBz
from freegs._forces import getForces

class MultiCoil(Coil):
    """
    This class is multifunctional and can model several coil arrangements:
    1. A single PF coil
    2. A block of several PF coil filaments, modelled as
        a) a single point in (R,Z)
        b) a list of (R,Z) points for each filament in the block

    In both cases, the specified coil(s) can also be mirrored in Z=0 and
    connected in a circuit sharing the same power supply. In such a case,
    there is also the option to specify the polarity of the currents in
    the upper and lower coil blocks, enabling the wiring of these blocks
    in opposite directions.

    public members
    --------------

    R, Z - Location of the point coil/Locations of coil filaments
    current - current in the coil(s) in Amps
    turns   - Number of turns if using point coils
    control - enable or disable control system
    mirror - enable or disable mirroring of coil block in Z=0
    polarity - wiring of coil blocks in circuit
    area    - Cross-section area in m^2

    For multiple point coils, the total toroidal current carried by the
    point coil block is current * turns
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
            (str("mirror"), bool),
            (str("polarity"), "2f8"),
        ]
    )

    def __init__(
        self,
        R,
        Z,
        current=0.0,
        turns=1,
        control=True,
        mirror=False,
        polarity=[1.0, 1.0],
        area=AreaCurrentLimit(),
    ):
        """
        R, Z       - Location of the coil centre. If modified moves all filaments
        Rfil, Zfil - Locations of coil filaments (lists)

        current - current in each turn of the coil in Amps
        turns   - Number of turns in point coil(s) block. Total block current is current * turns
                  This is only used if R,Z are a single point
        control - enable or disable control system
        mirror - mirror the point/detailed coil block in Z=0, creating a circuit
        polarity - Wiring of the circuit: same or opposite direction, [Block 1, Block 2]
        area    - Cross-section area of block in m^2

        Area can be a fixed value (e.g. 0.025 for 5x5cm coil), or can be specified
        using a function which takes a coil as an input argument.
        To specify a current density limit, use:

        area = AreaCurrentLimit(current_density)

        where current_density is in A/m^2. The area of the coil will be recalculated
        as the coil current is changed.

        The most important effect of the area is on the coil self-force:
        The smaller the area the larger the hoop force for a given current.
        """
        # Store locations as an internal list
        if hasattr(R, "__len__"):
            assert len(R) == len(Z)
            self.Rfil = R
            self.Zfil = Z
            self.turns = len(R)
        else:
            # Assume a single R, Z. Turn into a list
            self.Rfil = [R]
            self.Zfil = [Z]
            self.turns = turns

        self.current = current
        self.control = control
        self.mirror = mirror
        self.polarity = polarity
        self.area = area

        # Internal (R,Z) value, should not be modified directly
        self._R_centre = np.mean(self.Rfil)
        self._Z_centre = np.mean(self.Zfil)

    def controlPsi(self, R, Z):
        """
        Calculate poloidal flux at (R,Z) due to a unit current
        """

        result = 0.0
        for R_fil, Z_fil in zip(self.Rfil, self.Zfil):
            result += Greens(R_fil, Z_fil, R, Z) * self.polarity[0]

        if self.mirror:
            for R_fil, Z_fil in zip(self.Rfil, self.Zfil):
                result += Greens(R_fil, -Z_fil, R, Z) * self.polarity[1]
        return result

    def getCentreForces(self, equilibrium):
        """
        Calculate force acting on the coil calculated with respect to its centre.

        Return
        ------
        array [ Fr, Fz ] Radial (hoop) and vertical force.
        """
        return super(MultiCoil, self).getForces(equilibrium)

    def getForces(self, equilibrium):
        """
        Calculate total force acting on the coil.

        Return
        ------
        array [ Fr, Fz ] Radial (hoop) and vertical force.
        """
        fil_forces = self.getFilamentForces(equilibrium)
        mean_forces = np.sum(fil_forces, axis=1)

        return mean_forces

    def getFilamentForces(self, equilibrium):
        """
        Calculate force acting on each coil filament separately.

        Return
        ------
        array [ Fr: array, Fz: array) ] - Arrays of radial (hoop) and vertical forces.
        """
        return getForces(self.Rfil, self.Zfil, self, equilibrium)

    def controlBr(self, R, Z):
        """
        Calculate radial magnetic field Br at (R,Z) due to a unit current
        """
        result = 0.0
        for R_fil, Z_fil in zip(self.Rfil, self.Zfil):
            result += GreensBr(R_fil, Z_fil, R, Z) * self.polarity[0]

        if self.mirror:  # Mirror coil(s) in Z, creating circuit
            for R_fil, Z_fil in zip(self.Rfil, self.Zfil):
                result += GreensBr(R_fil, -Z_fil, R, Z) * self.polarity[1]
        return result

    def controlBz(self, R, Z):
        """
        Calculate vertical magnetic field Bz at (R,Z) due to a unit current
        """
        result = 0.0
        for R_fil, Z_fil in zip(self.Rfil, self.Zfil):
            result += GreensBz(R_fil, Z_fil, R, Z) * self.polarity[0]

        if self.mirror:  # Mirror coil(s) in Z, creating circuit
            for R_fil, Z_fil in zip(self.Rfil, self.Zfil):
                result += GreensBz(R_fil, -Z_fil, R, Z) * self.polarity[1]
        return result

    def __repr__(self):
        return "MultiCoil(R={0}, Z={1}, current={2:.1f}, turns={3}, control={4}, mirror={5}, polarity={6})".format(
            self.Rfil,
            self.Zfil,
            self.current,
            self.turns,
            self.control,
            self.mirror,
            self.polarity,
        )

    def __eq__(self, other):
        return (
            self.R == other.R
            and self.Z == other.Z
            and self.current == other.current
            and self.turns == other.turns
            and self.control == other.control
        )

    def __ne__(self, other):
        return not self == other

    def to_numpy_array(self):
        """
        Helper method for writing output
        """

        RZlen = len(self.Rfil)
        assert RZlen <= 500
        R = np.zeros(500)
        Z = np.zeros(500)
        R[:RZlen] = self.Rfil
        Z[:RZlen] = self.Zfil

        return np.array(
            (
                RZlen,
                R,
                Z,
                self.current,
                self.turns,
                self.control,
                self.mirror,
                self.polarity,
            ),
            dtype=self.dtype,
        )

    @classmethod
    def from_numpy_array(cls, value):
        if value.dtype != cls.dtype:
            raise ValueError(
                "Can't create {this} from dtype: {got} (expected: {dtype})".format(
                    this=type(cls), got=value.dtype, dtype=cls.dtype
                )
            )
        RZlen = value["RZlen"]
        R = value["R"][:RZlen]
        Z = value["Z"][:RZlen]
        current = value["current"]
        turns = value["turns"]
        control = value["control"]
        mirror = value["npoints"]
        polarity = value["polarity"]

        return MultiCoil(
            R,
            Z,
            current=current,
            turns=turns,
            control=control,
            mirror=mirror,
            polarity=polarity,
        )

    def plot(self, axis=None, show=False):
        """
        Plot the coil including turn locations
        """
        import matplotlib.pyplot as plt

        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)

        plt.plot(self.Rfil, self.Zfil, "bo")

        return axis

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
        for i in range(len(self.Rfil)):
            self.Rfil[i] += Rshift
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
        for i in range(len(self.Zfil)):
            self.Zfil[i] += Zshift
        self._Z_centre = Znew
