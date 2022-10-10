from freegs import geqdsk
from freegs import machine


def time_read_geqdsk():
    tokamak = machine.TestTokamak()

    with open("lsn.geqdsk") as f:
        geqdsk.read(f, tokamak, show=False)
