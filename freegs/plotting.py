"""
Plotting using matplotlib

Copyright 2016 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk

This file is part of FreeGS.

FreeGS is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FreeGS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGS.  If not, see <http://www.gnu.org/licenses/>.

"""

from numpy import linspace, amin, amax
from . import critical


def plotCoils(coils, axis=None):
    import matplotlib.pyplot as plt

    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot(111)

    return axis


def plotConstraints(control, axis=None, show=True):
    """
    Plots constraints used for coil current control

    axis     - Specify the axis on which to plot
    show     - Call matplotlib.pyplot.show() before returning

    """

    import matplotlib.pyplot as plt

    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot(111)

    # Locations of the X-points
    for r, z in control.xpoints:
        axis.plot(r, z, "bx")

    if control.xpoints:
        axis.plot([], [], "bx", label="X-point constraints")

    # Isoflux surfaces
    for r1, z1, r2, z2 in control.isoflux:
        axis.plot([r1, r2], [z1, z2], ":b^")

    if control.isoflux:
        axis.plot([], [], ":b^", label="Isoflux constraints")

    if show:
        plt.legend()
        plt.show()

    return axis


def plotEquilibrium(eq, axis=None, show=True, oxpoints=True, wall=True):
    """
    Plot the equilibrium flux surfaces

    axis     - Specify the axis on which to plot
    show     - Call matplotlib.pyplot.show() before returning
    oxpoints - Plot X points as red circles, O points as green circles
    wall     - Plot the wall (limiter)

    """

    import matplotlib.pyplot as plt

    R = eq.R
    Z = eq.Z
    psi = eq.psi()

    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot(111)

    levels = linspace(amin(psi), amax(psi), 100)

    axis.contour(R, Z, psi, levels=levels)
    axis.set_aspect("equal")
    axis.set_xlabel("Major radius [m]")
    axis.set_ylabel("Height [m]")

    if oxpoints:
        # Add O- and X-points
        opt, xpt = critical.find_critical(eq.R, eq.Z, psi)

        for r, z, _ in xpt:
            axis.plot(r, z, "ro")
        for r, z, _ in opt:
            axis.plot(r, z, "go")

        if xpt:
            psi_bndry = xpt[0][2]
            sep_contour = axis.contour(eq.R, eq.Z, psi, levels=[psi_bndry], colors="r")

            # Add legend
            axis.plot([], [], "ro", label="X-points")
            axis.plot([], [], "r", label="Separatrix")
        if opt:
            axis.plot([], [], "go", label="O-points")

    if wall and eq.tokamak.wall and len(eq.tokamak.wall.R):
        axis.plot(
            list(eq.tokamak.wall.R) + [eq.tokamak.wall.R[0]],
            list(eq.tokamak.wall.Z) + [eq.tokamak.wall.Z[0]],
            "k",
        )

    if show:
        plt.legend()
        plt.show()

    return axis
