from freegs import geqdsk
from freegs import machine
from freegs.plotting import plotEquilibrium

tokamak = machine.TestTokamak()

with open("lsn.geqdsk") as f:
    eq = geqdsk.read(f, tokamak, show=True)

plotEquilibrium(eq)
