from . import _divgeo
from . import geqdsk


def write(eq, fh, oxpoints=None):
    """
    Write a DivGeo equilibrium file, given a FreeGS Equilibrium object

    eq - Equilibrium object
    fh - file handle
    """

    geqdsk.write(eq, fh, oxpoints=oxpoints, fileformat=_divgeo.write)
