# Py2/3 compatibility
from __future__ import unicode_literals
from builtins import str

import h5py
import numpy as np

from .equilibrium import Equilibrium
from .machine import Coil, Circuit, Machine

string_dtype = h5py.special_dtype(vlen=str)

coil_dtype = np.dtype([
    (("R", "Major radius"), np.float64),
    (("Z", "Vertical position"), np.float64),
    (("current", "Current (Amps)"), np.float64),
    (("control", "Use feedback control?"), np.bool),
])

circuit_dtype = np.dtype([
    (("label", "Coil label"), string_dtype),
    (("R", "Major radius"), np.float64),
    (("Z", "Vertical position"), np.float64),
    (("current", "Current (Amps)"), np.float64),
    (("control", "Use feedback control?"), np.bool),
    (("multiplier", "Mupltiplication factor for current"), np.float64),
])

solenoid_dtype = np.dtype([
    (("Rs", "Radius"), np.float64),
    (("Zsmin", "Minimum Z"), np.float64),
    (("Zsmax", "Maximum Z"), np.float64),
    (("Ns", "Number of turns"), np.float64),
    (("current", "Current (Amps)"), np.float64),
    (("control", "Use feedback control?"), np.bool),
])


def dump_equilibrium(equilibrium, filename):
    with h5py.File(filename, 'w') as dumpfile:
        dumpfile["coil_dtype"] = coil_dtype
        coil_dtype_id = dumpfile["coil_dtype"]

        dumpfile["circuit_dtype"] = circuit_dtype
        circuit_dtype_id = dumpfile["circuit_dtype"]

        dumpfile["solenoid_dtype"] = solenoid_dtype
        solenoid_dtype_id = dumpfile["solenoid_dtype"]

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

        tokamak_group.create_dataset("wall_R", data=equilibrium.tokamak.wall.R)
        tokamak_group.create_dataset("wall_Z", data=equilibrium.tokamak.wall.Z)

        for label, coil in equilibrium.tokamak.coils:
            if isinstance(coil, Coil):
                coils_group.create_dataset(
                    label, dtype=coil_dtype_id,
                    data=np.array((coil.R, coil.Z, coil.current, coil.control),
                                  dtype=coil_dtype)
                )
            elif isinstance(coil, Solenoid):
                coils_group.create_dataset(
                    label, dtype=solenoid_dtype_id,
                    data=np.array((coil.Rs, coil.Zmin, coil.Zmax, coil.Ns,
                                   coil.current, coil.control), dtype=solenoid_dtype)
                )
            elif isinstance(coil, Circuit):
                circuit_id = coils_group.create_dataset(label, dtype=circuit_dtype_id,
                                                        shape=(len(coil.coils),))

                # Store the Circuit's current/control as attributes on
                # the dataset. The individual Coils in the Circuit
                # have the default values (0/False). A different way
                # of doing this might be to store the Circuit's
                # current/control in each Coil, and when reading, use
                # the values from the first Coil.
                circuit_dtype_id.attrs["current"] = coil.current
                circuit_dtype_id.attrs["control"] = coil.control

                for index, (sublabel, subcoil, multiplier) in enumerate(coil.coils):
                    circuit_id[index] = (sublabel, subcoil.R, subcoil.Z, 0.0,
                                         False, multiplier)
