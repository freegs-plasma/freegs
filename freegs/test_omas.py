import matplotlib.pyplot as plt
import numpy as np
import omas
import pytest

import freegs
from freegs.omasio import load_omas_coils, load_omas_machine, load_omas_circuits


# ODS feature with pf_active data
@pytest.fixture
def example_ods():
    ods = omas.ods_sample()

    r_coil = 1.3
    z_coil = -1.5
    dr_coil = 0.1
    dz_coil = 0.1
    n_elements = 9

    # Add a coil with 9 filaments:
    ods["pf_active.coil.+.name"] = "multicoil"

    for i in range(n_elements):
        icol = i % 3
        irow = i // 3
        rc = r_coil - 3 * dr_coil + irow * dr_coil
        zc = z_coil - 3 * dz_coil + icol * dz_coil

        ods["pf_active.coil.-1.element.+.geometry"]["rectangle"]["r"] = rc
        ods["pf_active.coil.-1.element.-1.geometry"]["rectangle"]["z"] = zc
        ods["pf_active.coil.-1.element.-1.geometry"]["rectangle"]["width"] = 0.1
        ods["pf_active.coil.-1.element.-1.geometry"]["rectangle"]["height"] = 0.1
        ods["pf_active.coil.-1.element.-1.geometry"]["geometry_type"] = 2

    # For the consistency with `example_circuit_ods` fixture:
    ods["pf_active.coil.-1.element.0.identifier"] = "sampl_circuit"

    return ods


@pytest.fixture
def example_circuit_ods(example_ods):
    ods = example_ods.copy()

    n_coils = len(ods["pf_active.coil"])

    # Generate Power Supplies:
    for coil_idx in ods["pf_active.coil"]:
        coil = ods["pf_active.coil"][coil_idx]
        ods["pf_active.supply.+.name"] = "PS_" + coil["element.0.identifier"]

    for coil_idx in ods["pf_active.coil"]:
        coil = ods["pf_active.coil"][coil_idx]
        # Add a circuit:
        ods["pf_active.circuit.+.name"] = "C_" + coil["element.0.identifier"]

        # Connect both poles in a loop
        connections = np.zeros((2, 4 * n_coils), dtype=int)
        # Connection on two poles
        connections[0, 2 * coil_idx] = 1
        connections[0, 2 * coil_idx + 2 * n_coils] = 1
        # Connection of the second two poles
        connections[1, 2 * coil_idx + 1] = 1
        connections[1, 2 * coil_idx + 2 * n_coils + 1] = 1

        ods["pf_active.circuit.-1.connections"] = connections

    return ods


def test_omas_coils(example_ods):
    coils = load_omas_coils(example_ods)

    coil_names = [coil[0] for coil in coils]

    assert "samp0" in coil_names
    assert "samp1" in coil_names
    assert "samp2" in coil_names


def test_omas_circuits(example_circuit_ods):
    circuits = load_omas_circuits(example_circuit_ods)

    circuit_names = [circuit[0] for circuit in circuits]

    assert "C_samp0" in circuit_names
    assert "C_samp1" in circuit_names
    assert "C_samp2" in circuit_names


def test_omas_machine(example_ods):
    machine = load_omas_machine(example_ods)


# Run only on request, since it takes a while to run
# @pytest.mark.skip(reason="Run only on request")
def test_omas_test_reconstruction(example_ods):
    machine = load_omas_machine(example_ods)

    eq = freegs.Equilibrium(machine,
                            Rmin=0.6, Rmax=2.8,
                            Zmin=-2.0, Zmax=1.8,
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
