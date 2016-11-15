from freegs import geqdsk
from freegs import machine
from freegs.plotting import plotEquilibrium

# Reading MAST equilibrium, up-down symmetric coils
tokamak = machine.MAST_sym()

#with open("g014220.00200") as f:
with open("mast.geqdsk") as f:
  eq = geqdsk.read(f, tokamak, show=True)

plotEquilibrium(eq)

