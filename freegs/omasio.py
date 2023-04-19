# IDS schema view: https://gafusion.github.io/omas/schema.html
from __future__ import annotations

from typing import Tuple, Type, List, Dict

import omas
import numpy as np

from freegs.coil import Coil
from .machine import ShapedCoil, FilamentCoil, Circuit

# OMAS coil types:
# multi-element coil is always translated to FilamentCoil (only support rectangular geometry)
# 'outline', 'rectangle' --> ShapeCoil
# oblique, arcs of circle, thick line and annulus are not supported at the moment
OMAS_COIL_TYPES = {1: 'outline',
                   2: 'rectangle',
                   3: 'oblique',
                   4: 'arcs of circle',
                   5: 'annulus',
                   6: 'thick line'}


def _identify_name(ods: omas.ODS) -> str | None:
    """ Identify coil name from OMAS data.
    Primary identifier is 'name', secondary is 'identifier'.
    Note: Not sure what is an intended difference between 'name' and 'identifier'.
    """
    if "name" in ods:
        return ods["name"]
    elif "identifier" in ods:
        return ods["identifier"]
    else:
        return None


def _load_omas_power_supplies(ods: omas.ODS) -> List[Dict]:
    """ Load power supplies names from OMAS data.
    :param ods: 'pf_active.power_supply.:' data"""
    # FreeGS does not have PowerSupply class, so we return data as list of dicts
    # The voltages can be loaded in the same way as currents to support equilibrium evolution.
    power_supplies_names = [{"name": _identify_name(ods[idx]),
                             "current": _load_omas_current(ods[idx])} for idx in ods]
    return power_supplies_names


def _identify_geometry_type(ods: omas.ODS) -> str | None:
    """ Identify coil geometry type from OMAS data.
    :param ods: pf_active.coil.element data
    :return: coil geometry type
    """
    geometry_type_idx = ods["geometry"]["geometry_type"] if "geometry" in ods else None
    geometry_type = OMAS_COIL_TYPES[geometry_type_idx] if geometry_type_idx else None

    if not geometry_type:
        if "outline" in ods["geometry"]:
            geometry_type = "outline"
        elif "rectangle" in ods["geometry"]:
            geometry_type = "rectangle"
        # The IDS definition of annulus is unclear to me.
        # elif "annulus" in ods["geometry"]:
        #     geometry_type = "annulus"
        else:
            geometry_type = None

    return geometry_type


def _load_omas_current(ods: omas.ODS) -> float:
    """ Load current from OMAS data.

    :param ods: any IDS substructure which can contain current structure data"""
    # Read current
    if "current" in ods:
        if len(ods["current"]["data"]) > 1:
            print(
                f"Warning: multiple circuit currents found. Using first one for time: "
                f"{ods['current']['time'][0]}")

        circuit_current = ods["current"]["data"][0]
    else:
        circuit_current = 0.0

    return circuit_current


def _circuit_connection_to_linear(circuit_structure: np.ndarray, n_supplies: int) -> Tuple[Tuple, Tuple]:
    num_rows, num_cols = circuit_structure.shape

    supply_idx = set()
    coil_idx = set()

    # Loop through each node in the circuit
    for i in range(num_rows):
        # Loop through each supply or coil side connected to the node
        for j in range(num_cols):
            if circuit_structure[i, j] == 1:
                # Determine if the connection is to a supply or coil
                if j < 2 * n_supplies:
                    index = j // 2
                    supply_idx.add(index)
                else:
                    index = (j - 2 * n_supplies) // 2
                    coil_idx.add(index)

    return tuple(supply_idx), tuple(coil_idx)


def _load_omas_circuit(ods: omas.ODS, coils: List[Tuple[str, Coil]], power_supplies: List[Dict]) -> Tuple[str, Circuit]:
    """ Load circuit from OMAS data.
    :param ods: 'pf_active.circuit.:' data"""
    # Identify circuit name

    # IDS circuit description can be found here: https://gafusion.github.io/omas/schema/schema_pf%20active.html.

    # Get linear circuit (coil and supply) structure
    supply_idx, coil_idx = _circuit_connection_to_linear(ods["connections"], len(power_supplies))

    if len(supply_idx) == 0 or len(coil_idx) == 0:
        raise ValueError(f"Invalid circuit structure. No supplies or coils found for circuit {ods}.")

    if len(supply_idx) > 1:
        print(f"Warning: multiple supplies found for circuit {ods}.")

    # Construct circuit name. First from circuit name, then from supply name, then from coil names.
    circuit_name = _identify_name(ods)
    if not circuit_name:
        if power_supplies[supply_idx[0]]["name"]:
            supply_names = [power_supplies[supply_idx[idx]]["name"] for idx in supply_idx if
                            power_supplies[supply_idx[idx]]["name"]]
            circuit_name = "+".join(supply_names)
        elif coils[coil_idx[0]][0]:
            coil_names = [coil[0] for coil in coils if coil[0]]
            circuit_name = "+".join(coil_names)
        else:
            raise ValueError(f"Unable to identify circuit name for circuit {ods}.")

    circuit_current = _load_omas_current(ods)

    # Init FreeGS circuit
    # TODO: Recognize correctly the multiplier for the circuit current
    circuit = Circuit([(coils[idx][0], coils[idx[1]], 1.0) for idx in coil_idx], circuit_current)
    return circuit_name, circuit


def _load_omas_circuits(ods: omas.ODS) -> List[Tuple[str, Circuit]]:
    coils = _load_omas_coils(ods)
    power_supplies = _load_omas_power_supplies(ods)
    return [_load_omas_circuit(ods["pf_active.circuit"][idx], coils, power_supplies) for idx in ods["pf_active.circuit"]]


def _load_omas_coil(ods: omas.ODS) -> Tuple[str, Coil]:
    """ Load coil from OMAS data.
    :param ods: 'pf_active.coil.:' data"""
    # Identify coil name
    coil_name = _identify_name(ods)

    # Read current
    coil_current = _load_omas_current(ods)

    # Multicoil or simple coil?
    if len(ods["element"]) > 1:
        r_filaments = ods["element"][:]["geometry"]["rectangle"]["r"]
        z_filaments = ods["element"][:]["geometry"]["rectangle"]["z"]
        turns = ods["element"][:]["turns_with_sign"]
        if not np.all(np.abs(turns) == 1):
            raise ValueError("Multicoil with non-unit turns is not supported yet.")

        # TODO: check if turns are interpreted correctly
        coil = FilamentCoil(r_filaments, z_filaments, coil_current, turns=len(r_filaments))
    else:
        print("Simple coil")
        if not coil_name:
            coil_name = _identify_name(ods["element"][0])

        if not coil_name:
            raise ValueError(f"Coil name not identified: \n {ods}")

        geometry_type = _identify_geometry_type(ods["element"][0])

        print(f"Coil name: {coil_name}")

        # Read turns:
        if "turns" in ods["element"][0]:
            turns = ods["element"][0]["turns_with_sign"]
        else:
            turns = 1

        # Init FreeGS coil
        if geometry_type == "outline":
            outline_r = ods["element"][0]["geometry"]["outline"]["r"]
            outline_z = ods["element"][0]["geometry"]["outline"]["z"]
            coil = ShapedCoil(list(zip(outline_r, outline_z)), coil_current, turns)
        elif geometry_type == "rectangle":
            r_centre = ods["element"][0]["geometry"]["rectangle"]["r"]
            z_centre = ods["element"][0]["geometry"]["rectangle"]["z"]
            width = ods["element"][0]["geometry"]["rectangle"]["width"]
            height = ods["element"][0]["geometry"]["rectangle"]["height"]
            coil_r = [r_centre - width / 2, r_centre + width / 2, r_centre + width / 2, r_centre - width / 2]
            coil_z = [z_centre - height / 2, z_centre - height / 2, z_centre + height / 2, z_centre + height / 2]
            coil = ShapedCoil(list(zip(coil_r, coil_z)), coil_current, turns)
        else:
            raise ValueError(f"Coil geometry type {geometry_type} not supported yet.")

    if coil_name is None:
        raise ValueError(f"Coil name not identified: \n {ods}")

    coil_tuple = (coil_name, coil)
    return coil_tuple


def _load_omas_coils(ods: omas.ODS) -> List[Tuple[str, Coil]]:
    coils = [_load_omas_coil(ods["pf_active.coil"][idx]) for idx in ods["pf_active.coil"]]
    return coils
