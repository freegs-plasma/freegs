# IDS schema view: https://gafusion.github.io/omas/schema.html
from typing import Tuple, Type, List

import omas
import numpy as np

from freegs.coil import Coil
from .machine import ShapedCoil, FilamentCoil

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
    if "name" in ods:
        return ods["name"]
    elif "identifier" in ods:
        return ods["identifier"]
    else:
        return None


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


def _load_omas_coil(ods: omas.ODS) -> Tuple[str, Type[Coil]]:
    """ Load coil from OMAS data.
    :param ods: 'pf_active.coil.:' data"""
    # Identify coil name
    coil_name = _identify_name(ods)

    # Read current
    if "current" in ods:
        if len(ods["current"]["data"]) > 1:
            print(
                f"Warning: multiple coil currents found. Using first one for time: "
                f"{ods['current']['time'][0]}")

        coil_current = ods["current"]["data"][0]
    else:
        coil_current = 0.0

    # Multicoil or simple coil?
    if len(ods["element"]) > 1:
        r_filaments = ods["element"][:]["geometry"]["rectangle"]["r"]
        z_filaments = ods["element"][:]["geometry"]["rectangle"]["z"]
        turns = ods["element"][:]["turns_with_sign"]
        if not np.all(np.abs(turns) == 1):
            raise ValueError("Multicoil with non-unit turns is not supported yet.")

        # TODO: check if turns are interpreted correctly
        coil = (coil_name, FilamentCoil(r_filaments, z_filaments, coil_current, turns=len(r_filaments)))
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
            coil = (coil_name, ShapedCoil(list(zip(outline_r, outline_z))), coil_current, turns)
        elif geometry_type == "rectangle":
            r_centre = ods["element"][0]["geometry"]["rectangle"]["r"]
            z_centre = ods["element"][0]["geometry"]["rectangle"]["z"]
            width = ods["element"][0]["geometry"]["rectangle"]["width"]
            height = ods["element"][0]["geometry"]["rectangle"]["height"]
            coil_r = [r_centre - width / 2, r_centre + width / 2, r_centre + width / 2, r_centre - width / 2]
            coil_z = [z_centre - height / 2, z_centre - height / 2, z_centre + height / 2, z_centre + height / 2]
            coil = (coil_name, ShapedCoil(list(zip(coil_r, coil_z))), coil_current, turns)
        else:
            raise ValueError(f"Coil geometry type {geometry_type} not supported yet.")
    return coil


def _load_omas_coils(ods: omas.ODS) -> List[Tuple[str, Type[Coil]]]:
    coils = [_load_omas_coil(ods["pf_active.coil"][idx]) for idx in ods["pf_active.coil"]]
    return coils
