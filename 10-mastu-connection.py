# Read MAST-U GEQDSK file, calculate connection length

from freegs import geqdsk
from freegs import machine

tokamak = machine.MASTU()

with open("mast-upgrade.geqdsk") as f:
    eq = geqdsk.read(f, tokamak, show=False)

from freegs import fieldtracer
import matplotlib.pyplot as plt

# Plot equilibrium
axis = eq.plot(show=False)

# Trace field lines both directions along the magnetic field
# By passing axis 
forward, backward = fieldtracer.traceFieldLines(eq, axis=axis, nturns=50)

plt.savefig("mast-upgrade-fieldtrace.pdf")
plt.show()

# Plot field line length from midplane to target
plt.plot(forward.R[0,:], forward.length[-1,:], label="Forward")
plt.plot(backward.R[0,:], backward.length[-1,:], label="Backward")
plt.legend()
plt.xlabel("Starting major radius [m]")
plt.ylabel("Parallel connection length [m]")

plt.savefig("mast-upgrade-lpar.pdf")
plt.show()
