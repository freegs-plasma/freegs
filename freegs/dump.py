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


class OutputFile(object):
    def __init__(self, name, mode=None, **kwds):
        self.handle = h5py.File(name, mode, **kwds)

    def close(self):
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def write_equilibrium(self, equilibrium):
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

        self.handle.create_dataset("Rmin", data=equilibrium.Rmin)
        self.handle.create_dataset("Rmax", data=equilibrium.Rmax)
        self.handle.create_dataset("R", data=equilibrium.R)

        self.handle.create_dataset("Zmin", data=equilibrium.Zmin)
        self.handle.create_dataset("Zmax", data=equilibrium.Zmax)
        self.handle.create_dataset("Z", data=equilibrium.Z)

        self.handle.create_dataset("current", data=equilibrium.plasmaCurrent())
        self.handle["current"].attrs["title"] = u"Plasma current [Amps]"
        self.handle.create_dataset("psi", data=equilibrium.psi())

        tokamak_group = self.handle.create_group("tokamak")
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

    def read_equilibrium(self):
        coil_dtype_id = self.handle["coil_dtype"]
        circuit_dtype_id = self.handle["circuit_dtype"]
        solenoid_dtype_id = self.handle["solenoid_dtype"]

        def make_coil(coil):
            return Coil(coil["R"], coil["Z"], coil["current"], coil["control"])

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

        def make_thing(thing):
            make_func = dtype_to_type[thing.dtype]
            return make_func(thing[()])

        coils = []
        for label, coil_ in self.handle["tokamak/coils"].items():
            coil = coil_[()]
            coils.append((label, make_thing(coil)))
        return coils
