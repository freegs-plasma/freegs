import freegs
from freegs import geqdsk
from freegs.equilibrium import refine
from freegs.plotting import plotEquilibrium

# Reading MAST equilibrium, up-down symmetric coils
tokamak = freegs.machine.MAST()

# with open("g014220.00200") as f:
with open("mast.geqdsk") as f:
    eq = geqdsk.read(f, tokamak, show=True)

# Increase resolution by a factor of 2
eq2 = refine(eq)

# Re-solve, keeping the same control points and profiles
freegs.solve(eq2, eq._profiles, eq.control, rtol=1e-6)

# Save to G-EQDSK
with open("mast-highres.geqdsk", "w") as f:
    geqdsk.write(eq2, f)

plotEquilibrium(eq2)
