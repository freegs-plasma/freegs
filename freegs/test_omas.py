import matplotlib.pyplot as plt
import numpy as np
import omas
import pytest

import freegs
from freegs.omasio import _load_omas_coils, load_omas_machine


# ODS feature with pf_active data
@pytest.fixture
def example_ods():
    ods = omas.ods_sample()

    ods["pf_active.coil.+.element"][0]["geometry"]["rectangle"]["r"] = 1.2
    ods["pf_active.coil.-1.element"][0]["geometry"]["rectangle"]["z"] = -1.5
    ods["pf_active.coil.-1.element"][0]["geometry"]["rectangle"]["width"] = 0.1
    ods["pf_active.coil.-1.element"][0]["geometry"]["rectangle"]["height"] = 0.1
    # set geometry_type = "rectangle"
    ods["pf_active.coil.-1.element"][0]["geometry"]["geometry_type"] = 2
    ods["pf_active.coil.-1.name"] = "test_coil"

    return ods


def test_omas_coils(example_ods):
    coils = _load_omas_coils(example_ods)

    coil_names = [coil[0] for coil in coils]

    assert "samp0" in coil_names
    assert "samp1" in coil_names
    assert "samp2" in coil_names


def test_omas_machine(example_ods):
    machine = load_omas_machine(example_ods)


# Run only on request, since it takes a while to run
@pytest.mark.skip(reason="Run only on request")
def test_omas_test_reconstruction(example_ods):
    machine = load_omas_machine(example_ods)

    eq = freegs.Equilibrium(machine,
                            Rmin=0.5, Rmax=3.0,
                            Zmin=-1.6, Zmax=1.5,
                            nx=65, ny=65)

    ip = example_ods["equilibrium"]["time_slice"][0]["global_quantities"]["ip"]
    p_ax = example_ods["equilibrium"]["time_slice"][0]["profiles_1d"]["pressure"][0]
    btor = example_ods["equilibrium"]["time_slice"][0]["global_quantities"]["magnetic_axis"]["b_field_tor"]
    rax = example_ods["equilibrium"]["time_slice"][0]["global_quantities"]["magnetic_axis"]["r"]
    f0 = btor * rax

    profiles = freegs.jtor.ConstrainPaxisIp(eq, p_ax, ip, f0)

    # Try some circular equilibrium
    r_lcfs = example_ods["equilibrium"]["time_slice"][0]["boundary"]["outline"]["r"]
    z_lcfs = example_ods["equilibrium"]["time_slice"][0]["boundary"]["outline"]["z"]

    isoflux = [(r_lcfs[0], z_lcfs[0], r, z) for r, z in zip(r_lcfs[1:], z_lcfs[1:])]
    constraints = freegs.control.constrain(isoflux=isoflux)

    freegs.solve(eq, profiles, constraints)

    ax = plt.gca()
    ax.set_aspect("equal")
    eq.plot(axis=ax, show=False)
    machine.plot(axis=ax, show=False)
    constraints.plot(axis=ax, show=False)
    plt.show()
