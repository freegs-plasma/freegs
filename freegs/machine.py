"""
Classes and routines to represent coils and circuits

License
-------

Copyright 2016-2019 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk

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

from .gradshafranov import Greens, GreensBr, GreensBz, mu0

from numpy import linspace
import numpy as np
import numbers

from .coil import Coil, AreaCurrentLimit
from .shaped_coil import ShapedCoil
from .multi_coil import MultiCoil

# We need this for the `label` part of the Circuit dtype for writing
# to HDF5 files. See the following for information:
# http://docs.h5py.org/en/latest/strings.html#how-to-store-text-strings
try:
    import h5py

    has_hdf5 = True
except ImportError:
    has_hdf5 = False


class Circuit:
    """
    Represents a collection of coils connected to the same circuit

    Public members
    --------------

    current  Current in the circuit [Amps]
    control  Use feedback control? [bool]
    """

    # We need a variable-length unicode string in the dtype. The numpy
    # 'S' dtype is not suitable, so we only fallback to it if we don't
    # have HDF5 (in which case, it doesn't matter, but we need
    # something)
    if has_hdf5:
        # Python 2/3 compatibility
        try:
            string_dtype = h5py.special_dtype(vlen=unicode)
        except NameError:
            string_dtype = h5py.special_dtype(vlen=str)
    else:
        string_dtype = np.dtype(str("S"))

    # A dtype for converting to Numpy array and storing in HDF5 files
    dtype = np.dtype(
        [
            (str("label"), string_dtype),
            (str("coil"), Coil.dtype),
            (str("multiplier"), np.float64),
        ]
    )

    def __init__(self, coils, current=0.0, control=True):
        """
        coils - A list [ (label, Coil, multiplier) ]
        """

        self.coils = coils
        self.current = current
        self.control = control

    def psi(self, R, Z):
        """
        Calculate poloidal flux at (R,Z)
        """
        psival = 0.0
        for label, coil, multiplier in self.coils:
            coil.current = self.current * multiplier
            psival += coil.psi(R, Z)
        return psival

    def createPsiGreens(self, R, Z):
        """
        Calculate Greens functions
        """
        pgreen = {}
        for label, coil, multiplier in self.coils:
            pgreen[label] = coil.createPsiGreens(R, Z)
        return pgreen

    def calcPsiFromGreens(self, pgreen):
        """
        Calculate psi from Greens functions

        """
        psival = 0.0
        for label, coil, multiplier in self.coils:
            coil.current = self.current * multiplier
            psival += coil.calcPsiFromGreens(pgreen[label])
        return psival

    def Br(self, R, Z):
        """
        Calculate radial magnetic field Br at (R,Z)
        """
        result = 0.0
        for label, coil, multiplier in self.coils:
            coil.current = self.current * multiplier
            result += coil.Br(R, Z)
        return result

    def Bz(self, R, Z):
        """
        Calculate vertical magnetic field Bz at (R,Z)
        """
        result = 0.0
        for label, coil, multiplier in self.coils:
            coil.current = self.current * multiplier
            result += coil.Bz(R, Z)
        return result

    def controlPsi(self, R, Z):
        """
        Calculate poloidal flux at (R,Z) due to a unit current
        """
        result = 0.0
        for label, coil, multiplier in self.coils:
            result += multiplier * coil.controlPsi(R, Z)
        return result

    def controlBr(self, R, Z):
        """
        Calculate radial magnetic field Br at (R,Z) due to a unit current
        """
        result = 0.0
        for label, coil, multiplier in self.coils:
            result += multiplier * coil.controlBr(R, Z)
        return result

    def controlBz(self, R, Z):
        """
        Calculate vertical magnetic field Bz at (R,Z) due to a unit current
        """
        result = 0.0
        for label, coil, multiplier in self.coils:
            result += multiplier * coil.controlBz(R, Z)
        return result

    def getForces(self, equilibrium):
        """
        Calculate forces on the coils

        Returns a dictionary of coil label -> force
        """
        forces = {}
        for label, coil, multiplier in self.coils:
            forces[label] = coil.getForces(equilibrium)
        return forces

    def __repr__(self):
        result = "Circuit(["
        coils = [
            '("{0}", {1}, {2})'.format(label, coil, multiplier)
            for label, coil, multiplier in self.coils
        ]
        result += ", ".join(coils)
        return result + "], current={0}, control={1})".format(
            self.current, self.control
        )

    def __eq__(self, other):
        return (
            self.coils == other.coils
            and self.current == other.current
            and self.control == other.control
        )

    def __ne__(self, other):
        return not self == other

    def to_numpy_array(self):
        """
        Helper method for writing output
        """
        return np.array(
            [
                (label, coil.to_numpy_array(), multiplier)
                for label, coil, multiplier in self.coils
            ],
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
        # Use the current/control values from the first coil in the circuit
        # Should be consistent!
        return Circuit(
            [(label, Coil(*coil), multiplier) for label, coil, multiplier in value],
            current=value[0][1]["current"] / value[0]["multiplier"],
            control=value[0][1]["control"],
        )

    def plot(self, axis=None, show=False):
        """
        Plot the coils in the circuit
        Returns the axis used
        """
        for label, coil, multiplier in self.coils:
            axis = coil.plot(axis=axis, show=False)
        if show:
            import matplotlib.pyplot

            plt.show()
        return axis


def MirroredCoil(
    R, Z, current=0.0, turns=1, control=True, area=AreaCurrentLimit(), symmetric=True
):
    """
    Create a pair of coils, at +/- Z
    If symmetric = True then current is in the same direction (in series);
    if symmetric = False then current is in the opposite direction
    """
    return Circuit(
        [
            (
                "U",
                Coil(R, Z, current=current, turns=turns, control=control, area=area),
                1.0,
            ),
            (
                "L",
                Coil(R, Z, current=current, turns=turns, control=control, area=area),
                1.0 if symmetric else -1.0,
            ),
        ]
    )


class Solenoid:
    """
    Represents a central solenoid

    Public members
    --------------

    current - current in each turn
    control - enable or disable control system

    """

    # A dtype for converting to Numpy array and storing in HDF5 files
    dtype = np.dtype(
        [
            (str("Rs"), np.float64),
            (str("Zsmin"), np.float64),
            (str("Zsmax"), np.float64),
            (str("Ns"), np.float64),
            (str("current"), np.float64),
            (str("control"), np.bool),
        ]
    )

    def __init__(self, Rs, Zsmin, Zsmax, Ns, current=0.0, control=True):
        """
        Rs - Radius of the solenoid
        Zsmin, Zsmax - Minimum and maximum Z
        Ns - Number of turns

        current - current in each turn
        control - enable or disable control system
        """
        self.Rs = Rs
        self.Zsmin = Zsmin
        self.Zsmax = Zsmax
        self.Ns = int(Ns)

        self.current = current
        self.control = control

    def psi(self, R, Z):
        """
        Calculate poloidal flux at (R,Z)
        """
        return self.controlPsi(R, Z) * self.current

    def createPsiGreens(self, R, Z):
        """
        Calculate Greens functions
        """
        return self.controlPsi(R, Z)

    def calcPsiFromGreens(self, pgreen):
        """
        Calculate psi from Greens functions
        """
        return self.current * pgreen

    def Br(self, R, Z):
        """
        Calculate radial magnetic field Br at (R,Z)
        """
        return self.controlBr(R, Z) * self.current

    def Bz(self, R, Z):
        """
        Calculate vertical magnetic field Bz at (R,Z)
        """
        return self.controlBz(R, Z) * self.current

    def controlPsi(self, R, Z):
        """
        Calculate poloidal flux at (R,Z) due to a unit current

        R and Z should have the same dimensions, but can be multi-dimensional
        Return should have the same shape
        """
        result = 0.0
        for Zs in linspace(self.Zsmin, self.Zsmax, self.Ns):
            result += Greens(self.Rs, Zs, R, Z)
        return result

    def controlBr(self, R, Z):
        """
        Calculate radial magnetic field Br at (R,Z) due to a unit current
        """
        result = 0.0
        for Zs in linspace(self.Zsmin, self.Zsmax, self.Ns):
            result += GreensBr(self.Rs, Zs, R, Z)
        return result

    def controlBz(self, R, Z):
        """
        Calculate vertical magnetic field Bz at (R,Z) due to a unit current
        """
        result = 0.0
        for Zs in linspace(self.Zsmin, self.Zsmax, self.Ns):
            result += GreensBz(self.Rs, Zs, R, Z)
        return result

    def getForces(self, equilibrium):
        """
        Calculate forces on the solenoid.
        Not currently implemented
        """
        return {}

    def __repr__(self):
        return "Solenoid(Rs={0}, Zsmin={1}, Zsmax={2}, current={3}, Ns={4}, control={5})".format(
            self.Rs, self.Zsmin, self.Zsmax, self.current, self.Ns, self.control
        )

    def __eq__(self, other):
        return (
            self.Rs == other.Rs
            and self.Zsmin == other.Zsmin
            and self.Zsmax == other.Zsmax
            and self.Ns == other.Ns
            and self.current == other.current
            and self.control == other.control
        )

    def __ne__(self, other):
        return not self == other

    def to_numpy_array(self):
        """
        Helper method for writing output
        """
        return np.array(
            (self.Rs, self.Zsmin, self.Zsmax, self.Ns, self.current, self.control),
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
        return Solenoid(*value[()])

    def plot(self, axis=None, show=False):
        return axis


class Wall:
    """
    Represents the wall of the device.
    Consists of an ordered list of (R,Z) points
    """

    def __init__(self, R, Z):
        self.R = R
        self.Z = Z

    def __repr__(self):
        return "Wall(R={R}, Z={Z})".format(R=self.R, Z=self.Z)

    def __eq__(self, other):
        return self.R == other.R and self.Z == other.Z

    def __ne__(self, other):
        return not self == other


class Machine:
    """
    Represents the machine (Tokamak), including
    coils and power supply circuits

    coils[(label, Coil|Circuit|Solenoid] - List of coils

    Note: a list is used rather than a dict, so that the coils
    remain ordered, and so can be updated easily by the control system.
    Instead __getitem__ is implemented to allow access to coils

    """

    def __init__(self, coils, wall=None):
        """
        coils - A list of coils [(label, Coil|Circuit|Solenoid)]
        """

        self.coils = coils
        self.wall = wall

    def __repr__(self):
        return "Machine(coils={coils}, wall={wall})".format(
            coils=self.coils, wall=self.wall
        )

    def __eq__(self, other):
        # Other Machine might be equivalent except for order of
        # coils. Assume this doesn't actually matter
        return sorted(self.coils) == sorted(other.coils) and self.wall == other.wall

    def __ne__(self, other):
        return not self == other

    def __getitem__(self, name):
        for label, coil in self.coils:
            if label == name:
                return coil
        raise KeyError("Machine does not contain coil with label '{0}'".format(name))

    def psi(self, R, Z):
        """
        Poloidal flux due to coils
        """
        psi_coils = 0.0
        for label, coil in self.coils:
            psi_coils += coil.psi(R, Z)

        return psi_coils

    def createPsiGreens(self, R, Z):
        """
        An optimisation, which pre-computes the Greens functions
        and puts into arrays for each coil. This map can then be
        called at a later time, and quickly return the field
        """
        pgreen = {}
        for label, coil in self.coils:
            pgreen[label] = coil.createPsiGreens(R, Z)
        return pgreen

    def calcPsiFromGreens(self, pgreen):
        """
        Uses the object returned by createPsiGreens to quickly
        compute the plasma psi
        """
        psi_coils = 0.0
        for label, coil in self.coils:
            psi_coils += coil.calcPsiFromGreens(pgreen[label])
        return psi_coils

    def Br(self, R, Z):
        """
        Radial magnetic field at given points
        """
        Br = 0.0
        for label, coil in self.coils:
            Br += coil.Br(R, Z)

        return Br

    def Bz(self, R, Z):
        """
        Vertical magnetic field
        """
        Bz = 0.0
        for label, coil in self.coils:
            Bz += coil.Bz(R, Z)

        return Bz

    def controlBr(self, R, Z):
        """
        Returns a list of control responses for Br
        at the given (R,Z) location(s).
        """
        return [coil.controlBr(R, Z) for label, coil in self.coils if coil.control]

    def controlBz(self, R, Z):
        """
        Returns a list of control responses for Bz
        at the given (R,Z) location(s)
        """
        return [coil.controlBz(R, Z) for label, coil in self.coils if coil.control]

    def controlPsi(self, R, Z):
        """
        Returns a list of control responses for psi
        at the given (R,Z) location(s)
        """
        return [coil.controlPsi(R, Z) for label, coil in self.coils if coil.control]

    def controlAdjust(self, current_change):
        """
        Add given currents to the controls.
        Given iterable must be the same length
        as the list returned by controlBr, controlBz
        """
        # Get list of coils being controlled
        controlcoils = [coil for label, coil in self.coils if coil.control]

        for coil, dI in zip(controlcoils, current_change):
            # Ensure that dI is a scalar
            coil.current += dI.item()

    def controlCurrents(self):
        """
        Return a list of coil currents for the coils being controlled
        """
        return [coil.current for label, coil in self.coils if coil.control]

    def setControlCurrents(self, currents):
        """
        Sets the currents in the coils being controlled.
        Input list must be of the same length as the list
        returned by controlCurrents
        """
        controlcoils = [coil for label, coil in self.coils if coil.control]
        for coil, current in zip(controlcoils, currents):
            coil.current = current

    def printCurrents(self):
        print("==========================")
        for label, coil in self.coils:
            print(label + " : " + str(coil))
        print("==========================")

    def getForces(self, equilibrium=None):
        """
        Calculate forces on the coils, given the plasma equilibrium.
        If no plasma equilibrium given then the forces due to
        the coils alone will be calculated.

        Returns a dictionary of coil label -> force
        """

        if equilibrium is None:
            equilibrium = self

        forces = {}
        for label, coil in self.coils:
            forces[label] = coil.getForces(equilibrium)
        return forces

    def getCurrents(self):
        """
        Returns a dictionary of coil label -> current in Amps
        """
        currents = {}
        for label, coil in self.coils:
            currents[label] = coil.current
        return currents

    def plot(self, axis=None, show=True):
        """
        Plot the machine coils
        """
        for label, coil in self.coils:
            axis = coil.plot(axis=axis, show=False)
        if show:
            import matplotlib.pyplot as plt

            plt.show()
        return axis


def EmptyTokamak():
    """
    Creates a tokamak with no coils
    """
    return Machine([])


def TestTokamak():
    """
    Create a simple tokamak
    """

    coils = [
        (
            "P1L",
            ShapedCoil([(0.95, -1.15), (0.95, -1.05), (1.05, -1.05), (1.05, -1.15)]),
        ),
        ("P1U", ShapedCoil([(0.95, 1.15), (0.95, 1.05), (1.05, 1.05), (1.05, 1.15)])),
        ("P2L", Coil(1.75, -0.6)),
        ("P2U", Coil(1.75, 0.6)),
    ]

    wall = Wall(
        [0.75, 0.75, 1.5, 1.8, 1.8, 1.5], [-0.85, 0.85, 0.85, 0.25, -0.25, -0.85]  # R
    )  # Z

    return Machine(coils, wall)


def DIIID():
    """
    PF coil set from ef20030203.d3d
    Taken from Corsica
    """

    coils = [
        {"label": "F1A", "R": 0.8608, "Z": 0.16830, "current": 0.0},
        {"label": "F2A", "R": 0.8614, "Z": 0.50810, "current": 0.0},
        {"label": "F3A", "R": 0.8628, "Z": 0.84910, "current": 0.0},
        {"label": "F4A", "R": 0.8611, "Z": 1.1899, "current": 0.0},
        {"label": "F5A", "R": 1.0041, "Z": 1.5169, "current": 0.0},
        {"label": "F6A", "R": 2.6124, "Z": 0.4376, "current": 0.0},
        {"label": "F7A", "R": 2.3733, "Z": 1.1171, "current": 0.0},
        {"label": "F8A", "R": 1.2518, "Z": 1.6019, "current": 0.0},
        {"label": "F9A", "R": 1.6890, "Z": 1.5874, "current": 0.0},
        {"label": "F1B", "R": 0.8608, "Z": -0.1737, "current": 0.0},
        {"label": "F2B", "R": 0.8607, "Z": -0.5135, "current": 0.0},
        {"label": "F3B", "R": 0.8611, "Z": -0.8543, "current": 0.0},
        {"label": "F4B", "R": 0.8630, "Z": -1.1957, "current": 0.0},
        {"label": "F5B", "R": 1.0025, "Z": -1.5169, "current": 0.0},
        {"label": "F6B", "R": 2.6124, "Z": -0.44376, "current": 0.0},
        {"label": "F7B", "R": 2.3834, "Z": -1.1171, "current": 0.0},
        {"label": "F8B", "R": 1.2524, "Z": -1.6027, "current": 0.0},
        {"label": "F9B", "R": 1.6889, "Z": -1.578, "current": 0.0},
    ]

    return Machine(coils)


def MAST():
    """
    Mega-Amp Spherical Tokamak. This version has all independent coils
    so that each is powered by a separate coil
    """

    coils = [
        ("P2U", Coil(0.49, 1.76)),
        ("P2L", Coil(0.49, -1.76)),
        ("P3U", Coil(1.1, 1.1)),
        ("P3L", Coil(1.1, -1.1)),
        ("P4U", Coil(1.51, 1.095)),
        ("P4L", Coil(1.51, -1.095)),
        ("P5U", Coil(1.66, 0.52)),
        ("P5L", Coil(1.66, -0.52)),
        ("P6U", Coil(1.5, 0.9)),
        ("P6L", Coil(1.5, -0.9)),
        ("P1", Solenoid(0.15, -1.4, 1.4, 100)),
    ]

    return Machine(coils)


def MAST_sym():
    """
    Mega-Amp Spherical Tokamak. This version the upper and lower coils
    are connected to the same circuits P2 - P6
    """
    coils = [
        (
            "P2",
            Circuit([("P2U", Coil(0.49, 1.76), 1.0), ("P2L", Coil(0.49, -1.76), 1.0)]),
        ),
        ("P3", Circuit([("P3U", Coil(1.1, 1.1), 1.0), ("P3L", Coil(1.1, -1.1), 1.0)])),
        (
            "P4",
            Circuit(
                [("P4U", Coil(1.51, 1.095), 1.0), ("P4L", Coil(1.51, -1.095), 1.0)]
            ),
        ),
        (
            "P5",
            Circuit([("P5U", Coil(1.66, 0.52), 1.0), ("P5L", Coil(1.66, -0.52), 1.0)]),
        ),
        ("P6", Circuit([("P6U", Coil(1.5, 0.9), 1.0), ("P6L", Coil(1.5, -0.9), -1.0)])),
        ("P1", Solenoid(0.15, -1.45, 1.45, 100)),
    ]

    return Machine(coils)


def TCV():
    coils = [
        #             ("A1", Coil(0.422500,0.000000)),
        #             ("B1", Coil(0.445700,-0.936000)),
        #             ("B2", Coil(0.445700,0.936000)),
        #             ("C1", Coil(0.621500,-1.110000)),
        #             ("C2", Coil(0.621500,1.110000)),
        #             ("D1", Coil(1.176500,-1.170000)),
        #             ("D2", Coil(1.176500,1.170000)),
        ("E1", Coil(0.505000, -0.700000)),
        ("E2", Coil(0.505000, -0.500000)),
        ("E3", Coil(0.505000, -0.300000)),
        ("E4", Coil(0.505000, -0.100000)),
        ("E5", Coil(0.505000, 0.100000)),
        ("E6", Coil(0.505000, 0.300000)),
        ("E7", Coil(0.505000, 0.500000)),
        ("E8", Coil(0.505000, 0.700000)),
        ("F1", Coil(1.309500, -0.770000)),
        ("F2", Coil(1.309500, -0.610000)),
        ("F3", Coil(1.309500, -0.310000)),
        ("F4", Coil(1.309500, -0.150000)),
        ("F5", Coil(1.309500, 0.150000)),
        ("F6", Coil(1.309500, 0.310000)),
        ("F7", Coil(1.309500, 0.610000)),
        ("F8", Coil(1.309500, 0.770000)),
        #             ("G1", Coil(1.098573,-0.651385)),
        #             ("G2", Coil(1.114000,-0.633000)),
        #             ("G3", Coil(1.129427,-0.614615)),
        #             ("G4", Coil(1.129427,0.614615)),
        #             ("G5", Coil(1.114000,0.633000)),
        #             ("G6", Coil(1.098573,0.651385)),
        ("T1", Coil(1.554000, -0.780000)),
        ("T2", Coil(1.717000, -0.780000)),
        ("T3", Coil(1.754000, -0.780000)),
        (
            "OH",
            Circuit(
                [
                    ("OH1", Solenoid(0.43, -0.93, 0.93, 100), 0.01),
                    ("C1", Coil(0.621500, -1.110000), 1.0),
                    ("C2", Coil(0.621500, 1.110000), 1.0),
                    ("D1", Coil(1.176500, -1.170000), 1.0),
                    ("D2", Coil(1.176500, 1.170000), 1.0),
                ]
            ),
        ),
    ]

    return Machine(coils)


def MASTU_simple():
    """This is an older version of the MAST-U coilset.
    A simplified set of coils, with one strand per coil.
    This may be easier to use for initial development of scenarios,
    but less detailed than the MultiCoil description (MASTU).
    """
    coils = [
        ("Solenoid", Solenoid(0.19475, -1.581, 1.581, 324)),
        # ("Pc", Coil(0.067, 0.0, turns=142)),
        ("Pc", Solenoid(0.067, -0.6, 0.6, 142)),
        (
            "Px",
            Circuit(
                [
                    ("PxU", Coil(0.2405, 1.2285, turns=44), 1.0),
                    ("PxL", Coil(0.2405, -1.2285, turns=44), 1.0),
                ]
            ),
        ),
        (
            "D1",
            Circuit(
                [
                    ("D1U", Coil(0.381, 1.555, turns=35), 1.0),
                    ("D1L", Coil(0.381, -1.555, turns=35), 1.0),
                ]
            ),
        ),
        (
            "D2",
            Circuit(
                [
                    ("D2U", Coil(0.574, 1.734, turns=23), 1.0),
                    ("D2L", Coil(0.574, -1.734, turns=23), 1.0),
                ]
            ),
        ),
        (
            "D3",
            Circuit(
                [
                    ("D3U", Coil(0.815, 1.980, turns=23), 1.0),
                    ("D3L", Coil(0.815, -1.980, turns=23), 1.0),
                ]
            ),
        ),
        (
            "Dp",
            Circuit(
                [
                    ("DpU", Coil(0.918, 1.501, turns=23), 1.0),
                    ("DpL", Coil(0.918, -1.501, turns=23), 1.0),
                ]
            ),
        ),
        (
            "D5",
            Circuit(
                [
                    ("D5U", Coil(1.900, 1.950, turns=27), 1.0),
                    ("D5L", Coil(1.900, -1.950, turns=27), 1.0),
                ]
            ),
        ),
        (
            "D6",
            Circuit(
                [
                    ("D6U", Coil(1.285, 1.470, turns=23), 1.0),
                    ("D6L", Coil(1.285, -1.470, turns=23), 1.0),
                ]
            ),
        ),
        (
            "D7",
            Circuit(
                [
                    ("D7U", Coil(1.520, 1.470, turns=23), 1.0),
                    ("D7L", Coil(1.520, -1.470, turns=23), 1.0),
                ]
            ),
        ),
        (
            "P4",
            Circuit(
                [
                    ("P4U", Coil(1.500, 1.100, turns=23), 1.0),
                    ("P4L", Coil(1.500, -1.100, turns=23), 1.0),
                ]
            ),
        ),
        (
            "P5",
            Circuit(
                [
                    ("P5U", Coil(1.650, 0.357, turns=23), 1.0),
                    ("P5L", Coil(1.650, -0.357, turns=23), 1.0),
                ]
            ),
        ),
        # Vertical control coils wired in opposite directions
        # Two pairs of coils, P6 pair 1 and P6 pair 2
        (
            "P61",
            Circuit(
                [
                    ("P61U", Coil(1.1975, 1.11175, turns=2), 1.0),
                    ("P61L", Coil(1.1975, -1.11175, turns=2), -1.0),
                ]
            ),
        ),
        (
            "P62",
            Circuit(
                [
                    ("P62U", Coil(1.2575, 1.0575, turns=2), 1.0),
                    ("P62L", Coil(1.2575, -1.0575, turns=2), -1.0),
                ]
            ),
        ),
    ]

    rwall = [
        1.6,
        1.2503,
        1.3483,
        1.47,
        1.47,
        1.45,
        1.45,
        1.3214,
        1.1904,
        0.89296,
        0.86938,
        0.83981,
        0.82229,
        0.81974,
        0.81974,
        0.82734,
        0.8548,
        0.89017,
        0.91974,
        0.94066,
        1.555,
        1.85,
        2,
        2,
        2,
        2,
        1.3188,
        1.7689,
        1.7301,
        1.35,
        1.09,
        1.09,
        0.90576,
        0.87889,
        0.87889,
        0.90717,
        0.53948,
        0.5112,
        0.50553,
        0.53594,
        0.5074,
        0.4974,
        0.5074,
        0.4788,
        0.4688,
        0.4788,
        0.333,
        0.333,
        0.275,
        0.334,
        0.261,
        0.261,
        0.244,
        0.261,
        0.261,
        0.244,
        0.261,
        0.261,
    ]

    zwall = [
        1,
        1,
        0.86,
        0.86,
        0.81,
        0.81,
        0.82,
        0.82,
        1.007,
        1.304,
        1.3312,
        1.3826,
        1.4451,
        1.4812,
        1.4936,
        1.5318,
        1.5696,
        1.5891,
        1.5936,
        1.5936,
        1.567,
        1.08,
        1.08,
        1.7,
        2.035,
        2.169,
        2.169,
        1.7189,
        1.68,
        2.06,
        2.06,
        2.06,
        1.8786,
        1.9055,
        1.9055,
        1.8772,
        1.5095,
        1.5378,
        1.5321,
        1.5017,
        1.4738,
        1.4838,
        1.4738,
        1.4458,
        1.4558,
        1.4458,
        1.303,
        1.1,
        1.1,
        1.1,
        0.502,
        0.348,
        0.348,
        0.348,
        0.146,
        0.146,
        0.146,
        0,
    ]

    # Concatenate with mirror image in Z,
    # with points in reverse order
    rwall = rwall + rwall[::-1]
    zwall = zwall + [-z for z in zwall[::-1]]

    return Machine(coils, Wall(rwall, zwall))


#########################################
# MAST-U, using MultiCoil to represent multiple strands


def MASTU():
    """MAST-Upgrade, using MultiCoil to represent coils with different locations
    for each strand.
    """
    d1_upper_r = [
        0.35275,
        0.36745015,
        0.38215014,
        0.39685005,
        0.35275,
        0.35275,
        0.35275,
        0.35275,
        0.35275039,
        0.36745039,
        0.36745039,
        0.36745039,
        0.36745039,
        0.36745,
        0.38215002,
        0.38215002,
        0.38215002,
        0.38215002,
        0.39685014,
        0.39685014,
        0.39685014,
        0.39685014,
        0.39685008,
        0.41155013,
        0.41155013,
        0.41155013,
        0.41155013,
        0.4115501,
        0.42625037,
        0.42625007,
        0.42625007,
        0.42625007,
        0.42625007,
        0.41155002,
        0.4262501,
    ]

    d1_upper_z = [
        1.60924995,
        1.60924995,
        1.60924995,
        1.60924995,
        1.59455001,
        1.57984996,
        1.5651499,
        1.55044997,
        1.53574991,
        1.53574991,
        1.55044997,
        1.5651499,
        1.57984996,
        1.59455001,
        1.57984996,
        1.5651499,
        1.55044997,
        1.53574991,
        1.53574991,
        1.55044997,
        1.5651499,
        1.57984996,
        1.59455001,
        1.59455001,
        1.57984996,
        1.5651499,
        1.55044997,
        1.53574991,
        1.53574991,
        1.55044997,
        1.5651499,
        1.57984996,
        1.59455001,
        1.60924995,
        1.60924995,
    ]

    d1_lower_z = [x * -1 for x in d1_upper_z]

    d2_upper_r = [
        0.60125011,
        0.58655024,
        0.60125017,
        0.60125017,
        0.60125023,
        0.58655,
        0.58655,
        0.57185012,
        0.57185036,
        0.57185042,
        0.55715007,
        0.55715007,
        0.55715001,
        0.54245019,
        0.54245019,
        0.54245001,
        0.52775019,
        0.52775025,
        0.52775025,
        0.57185012,
        0.55715013,
        0.54245007,
        0.52774996,
    ]

    d2_upper_z = [
        1.75705004,
        1.75705004,
        1.74234998,
        1.72765005,
        1.71294999,
        1.71294999,
        1.72765005,
        1.74234998,
        1.72765005,
        1.71294999,
        1.71294999,
        1.72765005,
        1.74234998,
        1.74234998,
        1.72765005,
        1.71294999,
        1.71294999,
        1.72765005,
        1.74234998,
        1.75705004,
        1.75705004,
        1.75705004,
        1.75705004,
    ]

    d2_lower_z = [x * -1 for x in d2_upper_z]

    d3_upper_r = [
        0.82854998,
        0.8432501,
        0.84325004,
        0.84325004,
        0.82855022,
        0.82855004,
        0.8285504,
        0.81384999,
        0.81385022,
        0.81385005,
        0.79915011,
        0.79915005,
        0.79915005,
        0.78445005,
        0.78444934,
        0.78445005,
        0.76975012,
        0.76975018,
        0.76975018,
        0.76975006,
        0.78445035,
        0.79915041,
        0.81384987,
    ]

    d3_upper_z = [
        2.00405002,
        2.00405002,
        1.98935008,
        1.97465003,
        1.95995009,
        1.97465003,
        1.98935008,
        1.98935008,
        1.97465003,
        1.95995009,
        1.95995009,
        1.97465003,
        1.98935008,
        1.98935008,
        1.97465003,
        1.95995009,
        1.95995009,
        1.97465003,
        1.98935008,
        2.00405002,
        2.00405002,
        2.00405002,
        2.00405002,
    ]

    d3_lower_z = [x * -1 for x in d3_upper_z]

    d5_upper_r = [
        1.90735006,
        1.92205048,
        1.92205,
        1.92205,
        1.92205,
        1.92205,
        1.92205,
        1.90735018,
        1.9073503,
        1.9073503,
        1.9073503,
        1.9073503,
        1.90735006,
        1.89265001,
        1.89265013,
        1.89265013,
        1.89265013,
        1.89265013,
        1.89265001,
        1.87794995,
        1.87795019,
        1.87795019,
        1.87795019,
        1.87795019,
        1.87795019,
        1.87795019,
        1.89265037,
    ]

    d5_upper_z = [
        1.99409997,
        1.99409997,
        1.97940004,
        1.96469998,
        1.95000005,
        1.93529999,
        1.92060006,
        1.9059,
        1.92060006,
        1.93529999,
        1.95000005,
        1.96469998,
        1.97940004,
        1.97940004,
        1.96469998,
        1.95000005,
        1.93529999,
        1.92060006,
        1.9059,
        1.9059,
        1.92060006,
        1.93529999,
        1.95000005,
        1.96469998,
        1.97940004,
        1.99409997,
        1.99409997,
    ]

    d5_lower_z = [x * -1 for x in d5_upper_z]

    d6_upper_r = [
        1.30704987,
        1.32175004,
        1.32175004,
        1.32174993,
        1.30705011,
        1.32174993,
        1.30704999,
        1.29235005,
        1.30704999,
        1.29235005,
        1.27765,
        1.27765,
        1.26295006,
        1.27765,
        1.26294994,
        1.24825013,
        1.26294994,
        1.24825001,
        1.24825001,
        1.24825013,
        1.26294994,
        1.27765,
        1.29234993,
    ]

    d6_upper_z = [
        1.44564998,
        1.44564998,
        1.46034992,
        1.47504997,
        1.48974991,
        1.48975003,
        1.47504997,
        1.46034992,
        1.46035004,
        1.47504997,
        1.48975003,
        1.47504997,
        1.46034992,
        1.46034992,
        1.47504997,
        1.48974991,
        1.48975003,
        1.47504997,
        1.46034992,
        1.44564998,
        1.44564998,
        1.44564998,
        1.44564998,
    ]

    d6_lower_z = [x * -1 for x in d6_upper_z]

    d7_upper_r = [
        1.54205,
        1.55675006,
        1.55675006,
        1.55675006,
        1.54205012,
        1.55674994,
        1.54205,
        1.52735007,
        1.54204988,
        1.52735007,
        1.51265013,
        1.52734995,
        1.51265001,
        1.49794996,
        1.51265001,
        1.49794996,
        1.48325002,
        1.48325002,
        1.48325002,
        1.48325002,
        1.49795008,
        1.51265001,
        1.52734995,
    ]

    d7_upper_z = [
        1.44564998,
        1.44564998,
        1.46034992,
        1.47504997,
        1.48974991,
        1.48974991,
        1.47504997,
        1.46034992,
        1.46034992,
        1.47504997,
        1.48974991,
        1.48974991,
        1.47504997,
        1.46035004,
        1.46034992,
        1.47504997,
        1.48974991,
        1.47504997,
        1.46035004,
        1.44564998,
        1.44564998,
        1.44564998,
        1.44564998,
    ]

    d7_lower_z = [x * -1 for x in d7_upper_z]

    dp_upper_r = [
        0.93285,
        0.94755,
        0.93285,
        0.94755,
        0.96224999,
        0.96224999,
        0.88875002,
        0.90345001,
        0.91815001,
        0.91815001,
        0.90345001,
        0.88875002,
        0.96224999,
        0.94755,
        0.93285,
        0.96224999,
        0.94755,
        0.93285,
        0.91815001,
        0.90345001,
        0.88875002,
        0.88874996,
        0.90345001,
        0.91815001,
    ]

    dp_upper_z = [
        1.48634994,
        1.48634994,
        1.47165,
        1.47165,
        1.48634994,
        1.47165,
        1.47165,
        1.47165,
        1.47165,
        1.48634994,
        1.48634994,
        1.48634994,
        1.51574993,
        1.51574993,
        1.51574993,
        1.50105,
        1.50105,
        1.50105,
        1.51574993,
        1.51574993,
        1.51574993,
        1.50105,
        1.50105,
        1.50105,
    ]

    dp_lower_z = [x * -1 for x in dp_upper_z]

    p4_upper_r = [
        1.43500018,
        1.53500021,
        1.51000023,
        1.48500025,
        1.46000016,
        1.43500006,
        1.43500006,
        1.46100008,
        1.43500018,
        1.46100008,
        1.48700011,
        1.4610002,
        1.48700011,
        1.51300013,
        1.48700011,
        1.51300013,
        1.53900015,
        1.51300013,
        1.53900003,
        1.56500018,
        1.53900015,
        1.56500006,
        1.56500006,
    ]

    p4_upper_z = [
        1.04014993,
        1.03714991,
        1.03714991,
        1.03714991,
        1.03714991,
        1.07814991,
        1.1161499,
        1.15414989,
        1.15414989,
        1.1161499,
        1.07814991,
        1.07814991,
        1.1161499,
        1.15414989,
        1.15414989,
        1.1161499,
        1.07814991,
        1.07814991,
        1.1161499,
        1.15414989,
        1.15414989,
        1.1161499,
        1.07814991,
    ]

    p4_lower_z = [x * -1 for x in p4_upper_z]

    p5_upper_r = [
        1.58500004,
        1.61000001,
        1.63499999,
        1.65999997,
        1.68499994,
        1.58500004,
        1.58500004,
        1.58500004,
        1.63499999,
        1.63499999,
        1.63499999,
        1.65999997,
        1.65999997,
        1.65999997,
        1.68499994,
        1.68500006,
        1.68500006,
        1.71500003,
        1.71500003,
        1.71500003,
        1.71500003,
        1.6099776,
        1.60997999,
    ]

    p5_upper_z = [
        0.41065004,
        0.41065004,
        0.41065004,
        0.41065004,
        0.41065004,
        0.37165004,
        0.33265004,
        0.29365003,
        0.37165004,
        0.33265004,
        0.29365003,
        0.29365003,
        0.33262005,
        0.37165004,
        0.37165004,
        0.33265004,
        0.29365003,
        0.29365006,
        0.33265004,
        0.37165004,
        0.41065004,
        0.31147972,
        0.35528255,
    ]

    p5_lower_z = [x * -1 for x in p5_upper_z]

    p6_upper_r = [
        1.2887001,
        1.2887001,
        1.30900013,
        1.2887001,
        1.30900013,
        1.33414996,
        1.33414996,
        1.35444999,
        1.33414996,
        1.35444999,
    ]

    p6_upper_z = [
        0.99616498,
        0.97586501,
        0.95556498,
        0.95556498,
        0.97586501,
        0.931265,
        0.91096503,
        0.89066499,
        0.89066499,
        0.91096503,
    ]

    p6_lower_z = [x * -1 for x in p6_upper_z]

    px_upper_r = [
        0.24849965,
        0.24849975,
        0.24849974,
        0.2344998,
        0.24849974,
        0.24849974,
        0.24849972,
        0.24849972,
        0.24849972,
        0.24849971,
        0.24849971,
        0.24849971,
        0.24849969,
        0.24849969,
        0.24849969,
        0.24849968,
        0.24849968,
        0.24849968,
        0.24849966,
        0.24849966,
        0.24849966,
        0.24849965,
        0.23449969,
        0.23449969,
        0.23449971,
        0.23449971,
        0.23449971,
        0.23449971,
        0.23449972,
        0.23449972,
        0.23449974,
        0.23449974,
        0.23449974,
        0.23449975,
        0.23449975,
        0.23449977,
        0.23449977,
        0.23449977,
        0.23449978,
        0.23449978,
        0.2344998,
        0.2344998,
    ]

    px_upper_z = [
        1.41640627,
        1.03640544,
        1.0554055,
        1.03164983,
        1.07440555,
        1.0934056,
        1.11240554,
        1.13140559,
        1.15040565,
        1.1694057,
        1.18840575,
        1.20740581,
        1.22640586,
        1.24540591,
        1.26440585,
        1.2834059,
        1.30240595,
        1.32140601,
        1.34040606,
        1.35940611,
        1.37840617,
        1.39740622,
        1.41164911,
        1.39264905,
        1.37364912,
        1.35464919,
        1.33564925,
        1.31664932,
        1.29764926,
        1.27864933,
        1.2596494,
        1.24064946,
        1.22164953,
        1.20264947,
        1.18364954,
        1.16464961,
        1.14564967,
        1.12664974,
        1.10764968,
        1.08864975,
        1.06964982,
        1.05064988,
    ]

    px_lower_z = [x * -1 for x in px_upper_z]

    pc_r = [
        0.05950115,
        0.05950115,
        0.05950116,
        0.05950116,
        0.05950116,
        0.05950121,
        0.05950117,
        0.05950117,
        0.05950118,
        0.05950122,
        0.05950119,
        0.05950119,
        0.05950119,
        0.05950119,
        0.05950123,
        0.0595012,
        0.0595012,
        0.0595012,
        0.0595012,
        0.0595012,
        0.0595012,
        0.0595012,
        0.0595012,
        0.0595012,
        0.0595012,
        0.0595012,
        0.0595012,
        0.0595012,
        0.05950124,
        0.05950119,
        0.05950119,
        0.05950119,
        0.05950119,
        0.05950119,
        0.05950122,
        0.05950122,
        0.05950118,
        0.05950117,
        0.05950117,
        0.05950117,
        0.05950116,
        0.05950116,
        0.05950115,
        0.05950115,
        0.05950114,
        0.05950114,
        0.05950113,
        0.05950113,
        0.05950112,
        0.05950112,
        0.05950111,
        0.0595011,
        0.05950113,
        0.05950109,
        0.05950108,
        0.05950111,
        0.0595011,
        0.05950109,
        0.05950109,
        0.05950107,
        0.0595009,
        0.05950106,
        0.05950104,
        0.05950104,
        0.05950103,
        0.05950105,
        0.05950146,
        0.05950177,
        0.05950198,
        0.05950211,
        0.05950096,
        0.07150049,
        0.0715005,
        0.07150051,
        0.07150052,
        0.07150053,
        0.07150054,
        0.07150055,
        0.07150055,
        0.07150057,
        0.07150058,
        0.07150058,
        0.07150059,
        0.0715006,
        0.07150061,
        0.07150061,
        0.07150062,
        0.07150063,
        0.07150064,
        0.07150064,
        0.07150065,
        0.07150066,
        0.07150067,
        0.07150067,
        0.07150067,
        0.07150068,
        0.07150069,
        0.0715007,
        0.0715007,
        0.0715007,
        0.07150071,
        0.07150071,
        0.07150072,
        0.07150066,
        0.07150073,
        0.07150073,
        0.07150073,
        0.07150073,
        0.07150074,
        0.07150074,
        0.07150085,
        0.07150132,
        0.07150154,
        0.07150155,
        0.07150079,
        0.07150076,
        0.07150076,
        0.07150076,
        0.07150076,
        0.07150076,
        0.07150076,
        0.07150076,
        0.07150073,
        0.07150077,
        0.07150077,
        0.07150077,
        0.07150077,
        0.07150077,
        0.07150077,
        0.07150076,
        0.07150076,
        0.07150076,
        0.07150076,
        0.07150076,
        0.07150076,
        0.07150076,
        0.07150076,
        0.07150076,
        0.07150076,
        0.07150075,
        0.07150075,
        0.07150074,
        0.0715007,
        0.07150044,
        0.07150045,
        0.07150047,
        0.07150047,
        0.07150049,
        0.07150128,
        0.07150129,
        0.07150111,
        0.07150067,
        0.07150006,
        0.07150054,
        0.07150055,
        0.07150055,
        0.07150056,
        0.07150057,
        0.07150058,
        0.07150058,
        0.0715006,
        0.07150061,
        0.07150061,
        0.07150062,
        0.07150062,
        0.07150063,
        0.07150064,
        0.07150064,
        0.07150065,
        0.07150065,
        0.07150066,
        0.07150067,
        0.07150067,
        0.07150067,
        0.07150068,
        0.07150068,
        0.07150069,
        0.07150069,
        0.0715007,
        0.0715007,
        0.0715007,
        0.0715007,
        0.0715007,
        0.07150071,
        0.07150071,
        0.07150071,
        0.07150072,
        0.07150072,
        0.07150072,
        0.07150072,
        0.07150072,
        0.07150072,
        0.07150073,
        0.07150073,
        0.07150073,
        0.07150073,
        0.07150073,
        0.07150073,
        0.07150073,
        0.07150073,
        0.07150073,
        0.07150076,
        0.07150076,
        0.07150076,
        0.07150076,
        0.07150076,
        0.07150076,
        0.07150076,
        0.07150076,
        0.07150075,
        0.07150075,
        0.07150075,
        0.07150074,
        0.059501,
        0.05950099,
        0.059501,
        0.05950118,
        0.05950101,
        0.05950101,
        0.05950101,
        0.05950102,
        0.05950102,
        0.05950102,
        0.05950103,
        0.05950122,
        0.05950122,
        0.05950122,
        0.05950104,
        0.05950104,
        0.05950104,
        0.05950104,
        0.05950104,
        0.05950104,
        0.05950104,
        0.05950104,
        0.05950104,
        0.0595013,
        0.05950121,
        0.05950123,
        0.05950116,
        0.05950124,
        0.05950121,
        0.05950119,
        0.05950121,
        0.05950122,
        0.05950122,
        0.05950121,
        0.0595012,
        0.05950121,
        0.05950121,
        0.0595012,
        0.05950101,
        0.05950119,
        0.05950119,
        0.05950118,
        0.05950118,
        0.05950117,
        0.05950117,
        0.05950116,
        0.05950116,
        0.05950119,
        0.05950115,
        0.05950114,
        0.05950114,
        0.05950113,
        0.05950112,
        0.05950112,
        0.05950111,
        0.0595011,
        0.05950109,
        0.05950108,
        0.05950107,
        0.05950107,
        0.05950106,
        0.05950105,
        0.05950104,
        0.05950103,
        0.05950102,
        0.05950101,
        0.059501,
        0.05950099,
        0.05950078,
        0.05950077,
        0.05950096,
    ]

    pc_z = [
        -7.81521082e-01,
        -7.59520829e-01,
        -7.37520635e-01,
        -7.15520382e-01,
        -6.93520188e-01,
        -6.71519935e-01,
        -6.49519742e-01,
        -6.27519488e-01,
        -6.05519295e-01,
        -5.83519042e-01,
        -5.61518848e-01,
        -5.39518654e-01,
        -5.17518401e-01,
        -4.95518208e-01,
        -4.73517984e-01,
        -4.51517761e-01,
        -4.29517567e-01,
        -4.07517344e-01,
        -3.85517120e-01,
        -3.63516897e-01,
        -3.41516674e-01,
        -3.19516450e-01,
        -2.97516227e-01,
        -2.75516003e-01,
        -2.53515780e-01,
        -2.31515557e-01,
        -2.09515333e-01,
        -1.87515110e-01,
        -1.65514901e-01,
        -1.43514678e-01,
        -1.21514454e-01,
        -9.95142311e-02,
        -7.75140151e-02,
        -5.55137955e-02,
        -3.35135758e-02,
        -1.15133552e-02,
        1.04868645e-02,
        3.24870832e-02,
        5.44873066e-02,
        7.64875263e-02,
        9.84877497e-02,
        1.20487973e-01,
        1.42488196e-01,
        1.64488405e-01,
        1.86488628e-01,
        2.08488852e-01,
        2.30489075e-01,
        2.52489269e-01,
        2.74489492e-01,
        2.96489716e-01,
        3.18489939e-01,
        3.40490162e-01,
        3.62490386e-01,
        3.84490609e-01,
        4.06490862e-01,
        4.28490698e-01,
        4.50491309e-01,
        4.72491503e-01,
        4.94491905e-01,
        5.16492069e-01,
        5.38492322e-01,
        5.60492516e-01,
        5.82492828e-01,
        6.04492962e-01,
        6.26493216e-01,
        6.48493409e-01,
        6.70493662e-01,
        6.92494035e-01,
        7.14494169e-01,
        7.36494422e-01,
        7.58494616e-01,
        7.58488476e-01,
        7.36488640e-01,
        7.14488804e-01,
        6.92489028e-01,
        6.70489192e-01,
        6.48489356e-01,
        6.26489520e-01,
        6.04489744e-01,
        5.82489908e-01,
        5.60490072e-01,
        5.38490295e-01,
        5.16490459e-01,
        4.94490594e-01,
        4.72490788e-01,
        4.50490981e-01,
        4.28491175e-01,
        4.06491339e-01,
        3.84491533e-01,
        3.62491727e-01,
        3.40491891e-01,
        3.18492085e-01,
        2.96492279e-01,
        2.74492443e-01,
        2.52492636e-01,
        2.30492830e-01,
        2.08493009e-01,
        1.86493203e-01,
        1.64493382e-01,
        1.42493561e-01,
        1.20493740e-01,
        9.84939337e-02,
        7.64944404e-02,
        5.44943213e-02,
        3.24945636e-02,
        1.04946950e-02,
        -1.15051102e-02,
        -3.35049182e-02,
        -5.55047244e-02,
        -7.75045305e-02,
        -9.95043367e-02,
        -1.21504150e-01,
        -1.43503949e-01,
        -1.65503755e-01,
        -1.87503561e-01,
        -2.09503368e-01,
        -2.31503174e-01,
        -2.53502995e-01,
        -2.75502801e-01,
        -2.97502607e-01,
        -3.19502413e-01,
        -3.41502219e-01,
        -3.63502026e-01,
        -3.85501832e-01,
        -4.07501638e-01,
        3.95501614e-01,
        -4.29501444e-01,
        -4.51501250e-01,
        -4.73501056e-01,
        -4.95500863e-01,
        -5.17500639e-01,
        -5.39500475e-01,
        -5.61500251e-01,
        -5.83500087e-01,
        -6.05499864e-01,
        -6.27499700e-01,
        -6.49499476e-01,
        -6.71499312e-01,
        -6.93499088e-01,
        -7.15498924e-01,
        -7.37498701e-01,
        -7.59498537e-01,
        -7.81498313e-01,
        7.69502759e-01,
        7.47502983e-01,
        7.25503147e-01,
        7.03503370e-01,
        6.81503534e-01,
        6.59503758e-01,
        6.37503922e-01,
        6.15504146e-01,
        5.93504310e-01,
        5.71504533e-01,
        5.49504697e-01,
        5.27504921e-01,
        5.05505085e-01,
        4.83505279e-01,
        4.61505353e-01,
        4.39505666e-01,
        4.17505413e-01,
        3.73506218e-01,
        3.51506412e-01,
        3.29506606e-01,
        3.07506770e-01,
        2.85506964e-01,
        2.63507158e-01,
        2.41507336e-01,
        2.19507530e-01,
        1.97507709e-01,
        1.75507888e-01,
        1.53508082e-01,
        1.31508261e-01,
        1.09508440e-01,
        8.75086263e-02,
        6.55088127e-02,
        4.35089879e-02,
        2.15091743e-02,
        -4.90644481e-04,
        -2.24904604e-02,
        -4.44902778e-02,
        -6.64900914e-02,
        -8.84899125e-02,
        -1.10489726e-01,
        -1.32489547e-01,
        -1.54489353e-01,
        -1.76489174e-01,
        -1.98488995e-01,
        -2.20488816e-01,
        -2.42488623e-01,
        -2.64488459e-01,
        -2.86488265e-01,
        -3.08488101e-01,
        -3.30487907e-01,
        -3.52487713e-01,
        -3.74487549e-01,
        -3.96487355e-01,
        -4.18487161e-01,
        -4.40486997e-01,
        -4.62486804e-01,
        -4.84486639e-01,
        -5.06486416e-01,
        -5.28486252e-01,
        -5.50486028e-01,
        -5.72485864e-01,
        -5.94485700e-01,
        -6.16485476e-01,
        -6.38485312e-01,
        -6.60485148e-01,
        -6.82484925e-01,
        -7.04484761e-01,
        -7.26484597e-01,
        -7.48484373e-01,
        -7.70484209e-01,
        -7.26499677e-01,
        -7.48499930e-01,
        -7.04499483e-01,
        -7.70499945e-01,
        -6.82499230e-01,
        -6.60498977e-01,
        -6.38498724e-01,
        -6.16498530e-01,
        -5.94498277e-01,
        -5.72498024e-01,
        -5.50497830e-01,
        -5.28497398e-01,
        -5.06497145e-01,
        -4.84496951e-01,
        -4.62496907e-01,
        -4.40496653e-01,
        -4.18496430e-01,
        -3.96496207e-01,
        -3.74495953e-01,
        -3.52495730e-01,
        -3.30495477e-01,
        -3.08495253e-01,
        -2.86495030e-01,
        -2.63993531e-01,
        -2.42494494e-01,
        -2.20494315e-01,
        -1.98494136e-01,
        -1.76493689e-01,
        -1.54493496e-01,
        -1.32493302e-01,
        -1.10493094e-01,
        -8.84926915e-02,
        -6.64924607e-02,
        -4.44922261e-02,
        -2.24919878e-02,
        -4.91753221e-04,
        2.15084739e-02,
        4.35086973e-02,
        6.55087605e-02,
        1.09509401e-01,
        1.31509617e-01,
        1.53509840e-01,
        1.75510064e-01,
        1.97510287e-01,
        2.19510511e-01,
        2.41510719e-01,
        2.63510942e-01,
        8.75091851e-02,
        2.85511166e-01,
        3.07511151e-01,
        3.29511374e-01,
        3.51511598e-01,
        3.73512030e-01,
        3.95512253e-01,
        4.17512476e-01,
        4.39512491e-01,
        4.61512923e-01,
        4.83513147e-01,
        5.05513191e-01,
        5.27513623e-01,
        5.49513638e-01,
        5.71513832e-01,
        5.93514264e-01,
        6.15514517e-01,
        6.37514710e-01,
        6.59514725e-01,
        6.81514919e-01,
        7.03515172e-01,
        7.25515604e-01,
        7.47515619e-01,
        7.69516051e-01,
    ]

    coils = [
        ("Solenoid", Solenoid(0.19475, -1.581, 1.581, 324, control=False)),
        ("Pc", MultiCoil(pc_r, pc_z)),
        (
            "Px",
            Circuit(
                [
                    ("PxU", MultiCoil(px_upper_r, px_upper_z), 1.0),
                    ("PxL", MultiCoil(px_upper_r, px_lower_z), 1.0),
                ]
            ),
        ),
        (
            "D1",
            Circuit(
                [
                    ("D1U", MultiCoil(d1_upper_r, d1_upper_z), 1.0),
                    ("D1L", MultiCoil(d1_upper_r, d1_lower_z), 1.0),
                ]
            ),
        ),
        (
            "D2",
            Circuit(
                [
                    ("D2U", MultiCoil(d2_upper_r, d2_upper_z), 1.0),
                    ("D2L", MultiCoil(d2_upper_r, d2_lower_z), 1.0),
                ]
            ),
        ),
        (
            "D3",
            Circuit(
                [
                    ("D3U", MultiCoil(d3_upper_r, d3_upper_z), 1.0),
                    ("D3L", MultiCoil(d3_upper_r, d3_lower_z), 1.0),
                ]
            ),
        ),
        (
            "Dp",
            Circuit(
                [
                    ("DPU", MultiCoil(dp_upper_r, dp_upper_z), 1.0),
                    ("DPL", MultiCoil(dp_upper_r, dp_lower_z), 1.0),
                ]
            ),
        ),
        (
            "D5",
            Circuit(
                [
                    ("D5U", MultiCoil(d5_upper_r, d5_upper_z), 1.0),
                    ("D5L", MultiCoil(d5_upper_r, d5_lower_z), 1.0),
                ]
            ),
        ),
        (
            "D6",
            Circuit(
                [
                    ("D6U", MultiCoil(d6_upper_r, d6_upper_z), 1.0),
                    ("D6L", MultiCoil(d6_upper_r, d6_lower_z), 1.0),
                ]
            ),
        ),
        (
            "D7",
            Circuit(
                [
                    ("D7U", MultiCoil(d7_upper_r, d7_upper_z), 1.0),
                    ("D7L", MultiCoil(d7_upper_r, d7_lower_z), 1.0),
                ]
            ),
        ),
        (
            "P4",
            Circuit(
                [
                    ("P4U", MultiCoil(p4_upper_r, p4_upper_z), 1.0),
                    ("P4L", MultiCoil(p4_upper_r, p4_lower_z), 1.0),
                ]
            ),
        ),
        (
            "P5",
            Circuit(
                [
                    ("P5U", MultiCoil(p5_upper_r, p5_upper_z), 1.0),
                    ("P5L", MultiCoil(p5_upper_r, p5_lower_z), 1.0),
                ]
            ),
        ),
        (
            "P6",
            Circuit(
                [
                    ("P6U", MultiCoil(p6_upper_r, p6_upper_z), 1.0),
                    ("P6L", MultiCoil(p6_upper_r, p6_lower_z), -1.0),
                ]
            ),
        ),
    ]

    rwall = [
        1.6,
        1.2503,
        1.3483,
        1.47,
        1.47,
        1.45,
        1.45,
        1.3214,
        1.1904,
        0.89296,
        0.86938,
        0.83981,
        0.82229,
        0.81974,
        0.81974,
        0.82734,
        0.8548,
        0.89017,
        0.91974,
        0.94066,
        1.555,
        1.85,
        2,
        2,
        2,
        2,
        1.3188,
        1.7689,
        1.7301,
        1.35,
        1.09,
        1.09,
        0.90576,
        0.87889,
        0.87889,
        0.90717,
        0.53948,
        0.5112,
        0.50553,
        0.53594,
        0.5074,
        0.4974,
        0.5074,
        0.4788,
        0.4688,
        0.4788,
        0.333,
        0.333,
        0.275,
        0.334,
        0.261,
        0.261,
        0.244,
        0.261,
        0.261,
        0.244,
        0.261,
        0.261,
    ]

    zwall = [
        1,
        1,
        0.86,
        0.86,
        0.81,
        0.81,
        0.82,
        0.82,
        1.007,
        1.304,
        1.3312,
        1.3826,
        1.4451,
        1.4812,
        1.4936,
        1.5318,
        1.5696,
        1.5891,
        1.5936,
        1.5936,
        1.567,
        1.08,
        1.08,
        1.7,
        2.035,
        2.169,
        2.169,
        1.7189,
        1.68,
        2.06,
        2.06,
        2.06,
        1.8786,
        1.9055,
        1.9055,
        1.8772,
        1.5095,
        1.5378,
        1.5321,
        1.5017,
        1.4738,
        1.4838,
        1.4738,
        1.4458,
        1.4558,
        1.4458,
        1.303,
        1.1,
        1.1,
        1.1,
        0.502,
        0.348,
        0.348,
        0.348,
        0.146,
        0.146,
        0.146,
        0,
    ]

    # Concatenate with mirror image in Z,
    # with points in reverse order
    rwall = rwall + rwall[::-1]
    zwall = zwall + [-z for z in zwall[::-1]]

    return Machine(coils, Wall(rwall, zwall))


if __name__ == "__main__":
    # Run test case

    # Define a machine with a single coil
    coils = [{"R": 2.0, "Z": "-1.5", "label": "P1", "current": 3.0}]
    tokamak = Machine(coils)
