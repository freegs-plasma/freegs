"""
Class for reading/writing freegs `Equilibrium` objects

Currently just HDF5 via h5py

License
-------

Copyright 2018 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk

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

# Py2/3 compatibility: h5py needs unicode for group/dataset names
from __future__ import unicode_literals

try:
    import h5py

    has_hdf5 = True
except ImportError:
    has_hdf5 = False

import numpy as np

from .equilibrium import Equilibrium
from .machine import Coil, Circuit, Solenoid, Wall, Machine
from . import boundary
from . import machine


class OutputFormatNotAvailableError(Exception):
    """Raised when we couldn't import HDF5 (or some other library)
    required for this OutputFile format

    """

    def __init__(self, file_format="HDF5"):
        self.message = "Sorry, {} is not available!".format(file_format)


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
        if not has_hdf5:
            raise OutputFormatNotAvailableError("HDF5")

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

        self.handle["coil_dtype"] = Coil.dtype
        self.handle["circuit_dtype"] = Circuit.dtype
        self.handle["solenoid_dtype"] = Solenoid.dtype

        type_to_dtype = {
            Coil.dtype: self.handle["coil_dtype"],
            Circuit.dtype: self.handle["circuit_dtype"],
            Solenoid.dtype: self.handle["solenoid_dtype"],
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
        equilibrium_group["current"].attrs["title"] = "Plasma current [Amps]"

        psi_id = equilibrium_group.create_dataset("psi", data=equilibrium.psi())
        psi_id.dims[0].label = "R"
        psi_id.dims[1].label = "Z"
        psi_id.dims.create_scale(equilibrium_group["R_1D"], "R")
        psi_id.dims.create_scale(equilibrium_group["Z_1D"], "Z")
        psi_id.dims[0].attach_scale(equilibrium_group["R_1D"])
        psi_id.dims[1].attach_scale(equilibrium_group["Z_1D"])

        plasma_psi_id = equilibrium_group.create_dataset(
            "plasma_psi", data=equilibrium.plasma_psi
        )
        plasma_psi_id.dims[0].label = "R"
        plasma_psi_id.dims[1].label = "Z"
        plasma_psi_id.dims[0].attach_scale(equilibrium_group["R_1D"])
        plasma_psi_id.dims[1].attach_scale(equilibrium_group["Z_1D"])

        equilibrium_group.create_dataset(
            "boundary_function", data=equilibrium._applyBoundary.__name__
        )

        tokamak_group = equilibrium_group.create_group(self.MACHINE_GROUP_NAME)

        if equilibrium.tokamak.wall is not None:
            tokamak_group.create_dataset("wall_R", data=equilibrium.tokamak.wall.R)
            tokamak_group.create_dataset("wall_Z", data=equilibrium.tokamak.wall.Z)

        coils_group = tokamak_group.create_group(self.COILS_GROUP_NAME)
        for label, coil in equilibrium.tokamak.coils:
            dtype = type_to_dtype[coil.dtype]
            coils_group.create_dataset(
                label, dtype=dtype, data=np.array(coil.to_numpy_array())
            )
            # A bit gross, but store the class name so we know what
            # type to restore it to later
            coils_group[label].attrs["freegs type"] = coil.__class__.__name__

    def read_equilibrium(self):
        """
        Read an equilibrium from the file

        Returns
        -------
        Equilibrium
            A new `Equilibrium` object
        """

        equilibrium_group = self.handle[self.EQUILIBRIUM_GROUP_NAME]
        tokamak_group = equilibrium_group[self.MACHINE_GROUP_NAME]
        coil_group = tokamak_group[self.COILS_GROUP_NAME]

        # This is also a bit hacky - find the appropriate class in
        # freegs.machine and then call the `from_numpy_array` class
        # method
        def make_coil_set(thing):
            return machine.__dict__[thing.attrs["freegs type"]].from_numpy_array(thing)

        # Unfortunately this creates the coils in lexographical order
        # by label, losing the origin
        coils = [(label, make_coil_set(coil)) for label, coil in coil_group.items()]

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

        equilibrium = Equilibrium(
            tokamak=tokamak,
            Rmin=Rmin,
            Rmax=Rmax,
            Zmin=Zmin,
            Zmax=Zmax,
            nx=nx,
            ny=ny,
            psi=plasma_psi,
            current=current,
            boundary=eq_boundary_func,
        )

        return equilibrium
