import omas

from freegs.omasio import _load_omas_coils


# ODS feature with pf_active data
def coils_ods():
    ods = omas.ods_sample()
    # TODO add multi-element coil
    return ods


def test_omas_coils():
    ods = coils_ods()

    coils = _load_omas_coils(ods)
    print(coils)
