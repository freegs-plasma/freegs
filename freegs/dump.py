# Py2/3 compatibility
from __future__ import unicode_literals
from builtins import str

import h5py
import numpy as np

from .equilibrium import Equilibrium
from .machine import Coil, Circuit, Solenoid, Machine

# Define these dtypes here in order to avoid having the string_dtype,
# which requires h5py, in the machine module. We need string_dtype
# because the Coils in Circuits have string labels.
string_dtype = h5py.special_dtype(vlen=str)

coil_dtype = np.dtype([
    ("R", np.float64),
    ("Z", np.float64),
    ("current", np.float64),
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


def dump_equilibrium(equilibrium, filename):
    with h5py.File(filename, 'w') as dumpfile:
        dumpfile["coil_dtype"] = coil_dtype
        coil_dtype_id = dumpfile["coil_dtype"]

        dumpfile["circuit_dtype"] = circuit_dtype
        circuit_dtype_id = dumpfile["circuit_dtype"]

        dumpfile["solenoid_dtype"] = solenoid_dtype
        solenoid_dtype_id = dumpfile["solenoid_dtype"]

        type_to_dtype = {
            Coil: coil_dtype_id,
            Circuit: circuit_dtype_id,
            Solenoid: solenoid_dtype_id,
        }

        dumpfile.create_dataset("Rmin", data=equilibrium.Rmin)
        dumpfile.create_dataset("Rmax", data=equilibrium.Rmax)
        dumpfile.create_dataset("R", data=equilibrium.R)

        dumpfile.create_dataset("Zmin", data=equilibrium.Zmin)
        dumpfile.create_dataset("Zmax", data=equilibrium.Zmax)
        dumpfile.create_dataset("Z", data=equilibrium.Z)

        dumpfile.create_dataset("current", data=equilibrium.plasmaCurrent())
        dumpfile["current"].attrs["title"] = u"Plasma current [Amps]"
        dumpfile.create_dataset("psi", data=equilibrium.psi())

        tokamak_group = dumpfile.create_group("tokamak")
        coils_group = tokamak_group.create_group("coils")

        if equilibrium.tokamak.wall is not None:
            tokamak_group.create_dataset(
                "wall_R", data=equilibrium.tokamak.wall.R)
            tokamak_group.create_dataset(
                "wall_Z", data=equilibrium.tokamak.wall.Z)

        for label, coil in equilibrium.tokamak.coils:
            try:
                shape = (len(coil),)
            except TypeError:
                shape = (1,)

            dtype = type_to_dtype[type(coil)]

            coils_group.create_dataset(label, dtype=dtype,
                                       data=np.array(coil.to_tuple(), dtype=dtype))
