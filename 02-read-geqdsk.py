from freegs import geqdsk, machine
from freegs.plotting import plotEquilibrium

tokamak = machine.TestTokamak()

with open("lsn.geqdsk") as f:
    eq = geqdsk.read(f, tokamak, show=True)

plotEquilibrium(eq)
