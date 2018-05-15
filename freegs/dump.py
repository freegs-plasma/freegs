# Py2/3 compatibility
from __future__ import unicode_literals
from builtins import str

import h5py
import numpy as np

from .equilibrium import Equilibrium
from .machine import Coil, Circuit, Solenoid, Wall, Machine
from . import boundary

# Define these dtypes here in order to avoid having the string_dtype,
# which requires h5py, in the machine module. We need string_dtype
# because the Coils in Circuits have string labels.
string_dtype = h5py.special_dtype(vlen=str)

coil_dtype = np.dtype([
    ("R", np.float64),
    ("Z", np.float64),
    ("current", np.float64),
    ("turns", np.int),
    ("control", np.bool),
])

circuit_dtype = np.dtype([
    ("label", string_dtype),
    ("coil", coil_dtype),
    ("multiplier", np.float64),
])

solenoid_dtype = np.dtype([
    ("Rs", np.float64),
    ("Zsmin", np.float64),
    ("Zsmax", np.float64),
    ("Ns", np.float64),
    ("current", np.float64),
    ("control", np.bool),
])


class OutputFile(object):
    """
    Read/write freegs Equilibrium objects to file

    Currently supports HDF5 format only

    Given an Equilibrium object, eq, write to file like:

    >>> with freegs.OutputFile("test_readwrite.h5", 'w') as f:
    ...     f.write_equilibrium(eq)

    Read back into an Equilibrium like so:

    >>> with freegs.OutputFile("test_readwrite.h5", 'r') as f:
    ...      eq = f.read_equilibrium()

    Parameters
    ----------
    name : str
           Name of file to open
    mode : str
           Mode string to pass to `h5py.File`, one of:
             r  - Read-only, file must exist
             r+ - Read-write, file must exist
             w  - Create file, truncate if exists
             w- - Create file, fail if exists
             a  - Read-write if exists, create otherwise
    **kwds
           Other keyword arguments to pass to `h5py.File`
    """

    # Names of the groups in the file
    EQUILIBRIUM_GROUP_NAME = "equilbrium"
    MACHINE_GROUP_NAME = "tokamak"
    COILS_GROUP_NAME = "coils"

    def __init__(self, name, mode=None, **kwds):
        self.handle = h5py.File(name, mode, **kwds)

    def close(self):
        """
        Close the file
        """
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def write_equilibrium(self, equilibrium):
        """
        Write `equilbrium` to file
        """

        self.handle["coil_dtype"] = coil_dtype
        coil_dtype_id = self.handle["coil_dtype"]

        self.handle["circuit_dtype"] = circuit_dtype
        circuit_dtype_id = self.handle["circuit_dtype"]

        self.handle["solenoid_dtype"] = solenoid_dtype
        solenoid_dtype_id = self.handle["solenoid_dtype"]

        type_to_dtype = {
            Coil: coil_dtype_id,
            Circuit: circuit_dtype_id,
            Solenoid: solenoid_dtype_id,
        }

        equilibrium_group = self.handle.require_group(self.EQUILIBRIUM_GROUP_NAME)

        equilibrium_group.create_dataset("Rmin", data=equilibrium.Rmin)
        equilibrium_group.create_dataset("Rmax", data=equilibrium.Rmax)
        equilibrium_group.create_dataset("R_1D", data=equilibrium.R_1D)
        equilibrium_group.create_dataset("R", data=equilibrium.R)

        equilibrium_group.create_dataset("Zmin", data=equilibrium.Zmin)
        equilibrium_group.create_dataset("Zmax", data=equilibrium.Zmax)
        equilibrium_group.create_dataset("Z_1D", data=equilibrium.Z_1D)
        equilibrium_group.create_dataset("Z", data=equilibrium.Z)

        equilibrium_group.create_dataset("current", data=equilibrium.plasmaCurrent())
        equilibrium_group["current"].attrs["title"] = u"Plasma current [Amps]"

        psi_id = equilibrium_group.create_dataset("psi", data=equilibrium.psi())
        psi_id.dims[0].label = "R"
        psi_id.dims[1].label = "Z"
        psi_id.dims.create_scale(equilibrium_group["R_1D"], "R")
        psi_id.dims.create_scale(equilibrium_group["Z_1D"], "Z")
        psi_id.dims[0].attach_scale(equilibrium_group["R_1D"])
        psi_id.dims[1].attach_scale(equilibrium_group["Z_1D"])

        plasma_psi_id = equilibrium_group.create_dataset("plasma_psi",
                                                         data=equilibrium.plasma_psi)
        plasma_psi_id.dims[0].label = "R"
        plasma_psi_id.dims[1].label = "Z"
        plasma_psi_id.dims[0].attach_scale(equilibrium_group["R_1D"])
        plasma_psi_id.dims[1].attach_scale(equilibrium_group["Z_1D"])

        equilibrium_group.create_dataset("boundary_function",
                                         data=equilibrium._applyBoundary.__name__)

        tokamak_group = equilibrium_group.create_group(self.MACHINE_GROUP_NAME)

        if equilibrium.tokamak.wall is not None:
            tokamak_group.create_dataset(
                "wall_R", data=equilibrium.tokamak.wall.R)
            tokamak_group.create_dataset(
                "wall_Z", data=equilibrium.tokamak.wall.Z)

        coils_group = tokamak_group.create_group(self.COILS_GROUP_NAME)
        for label, coil in equilibrium.tokamak.coils:
            dtype = type_to_dtype[type(coil)]
            coils_group.create_dataset(label, dtype=dtype,
                                       data=np.array(coil.to_tuple(), dtype=dtype))

    def read_equilibrium(self):
        """
        Read an equilibrium from the file

        Returns
        -------
        Equilibrium
            A new `Equilibrium` object
        """

        def make_coil(coil):
            return Coil(coil["R"], coil["Z"], coil["current"], coil["turns"], coil["control"])

        def make_circuit(circuit):
            return Circuit([(label, make_coil(coil), multiplier)
                            for label, coil, multiplier in circuit],
                           current=circuit[0][1]["current"] / circuit[0]["multiplier"],
                           control=circuit[0][1]["control"])

        def make_solenoid(solenoid):
            return Solenoid(solenoid["Rs"], solenoid["Zsmin"], solenoid["Zsmax"],
                            solenoid["Ns"], solenoid["current"], solenoid["control"])

        dtype_to_type = {
            coil_dtype: make_coil,
            circuit_dtype: make_circuit,
            solenoid_dtype: make_solenoid,
        }

        def make_coil_set(thing):
            make_func = dtype_to_type[thing.dtype]
            return make_func(thing[()])

        equilibrium_group = self.handle[self.EQUILIBRIUM_GROUP_NAME]
        tokamak_group = equilibrium_group[self.MACHINE_GROUP_NAME]
        coil_group = tokamak_group[self.COILS_GROUP_NAME]

        # Unfortunately this creates the coils in lexographical order
        # by label, losing the origin
        coils = [(label, make_coil_set(coil[()])) for label, coil in coil_group.items()]

        if "wall_R" in tokamak_group:
            wall_R = tokamak_group["wall_R"][:]
            wall_Z = tokamak_group["wall_Z"][:]
            wall = Wall(wall_R, wall_Z)
        else:
            wall = None

        tokamak = Machine(coils, wall)

        Rmin = equilibrium_group["Rmin"][()]
        Rmax = equilibrium_group["Rmax"][()]
        Zmin = equilibrium_group["Zmin"][()]
        Zmax = equilibrium_group["Zmax"][()]
        nx, ny = equilibrium_group["R"].shape

        current = equilibrium_group["current"][()]
        plasma_psi = equilibrium_group["plasma_psi"][()]

        # Feels a bit hacky... the boundary function is saved as a
        # string of the function __name__, which we then look up in
        # the boundary module dict
        eq_boundary_name = equilibrium_group["boundary_function"][()]
        eq_boundary_func = boundary.__dict__[eq_boundary_name]

        equilibrium = Equilibrium(tokamak=tokamak, Rmin=Rmin, Rmax=Rmax,
                                  Zmin=Zmin, Zmax=Zmax, nx=nx, ny=ny,
                                  psi=plasma_psi, current=current,
                                  boundary=eq_boundary_func)

        return equilibrium
