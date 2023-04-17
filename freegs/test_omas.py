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

    coil_names = [coil[0] for coil in coils]

    assert "samp0" in coil_names
    assert "samp1" in coil_names
    assert "samp2" in coil_names
