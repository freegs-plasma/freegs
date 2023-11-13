"""
Routines for solving the nonlinear part of the Grad-Shafranov equation

Copyright 2016-2019 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk

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

from numpy import amin, amax, array
import numpy as np


def solve(
    eq,
    profiles,
    constrain=None,
    rtol=1e-3,
    atol=1e-10,
    blend=0.0,
    show=False,
    axis=None,
    pause=0.0001,
    psi_bndry=None,
    maxits=50,
    convergenceInfo=False,
    check_limited=False,
    wait_for_limited=False,
    limit_it=0,
):
    """
    Perform Picard iteration to find solution to the Grad-Shafranov equation

    eq       - an Equilibrium object (equilibrium.py)
    profiles - A Profile object for toroidal current (jtor.py)

    rtol     - Relative tolerance (change in psi)/( max(psi) - min(psi) )
    atol     - Absolute tolerance, change in psi
    blend    - Blending of previous and next psi solution
               psi{n+1} <- psi{n+1} * (1-blend) + blend * psi{n}

    show     - If true, plot the plasma equilibrium at each nonlinear step
    axis     - Specify a figure to plot onto. Default (None) creates a new figure
    pause    - Delay between output plots. If negative, waits for window to be closed

    maxits   - Maximum number of iterations. Set to None for no limit.
               If this limit is exceeded then a RuntimeError is raised.

    convergenceInfo - True/False toggle for outputting convergence data.

    check_limited - True/False toggle to control checking for limited plasmas.
    wait_for_limited - True/False toggle to keep iterating until the plasma is limited.
    limit_it - Integer > Sometimes waiting some number of interations before checking if
    a plasma is limited can improve convergence. This sets the number of iterations to wait.
    """

    if constrain is not None:
        # Set the coil currents to get X-points in desired locations
        constrain(eq)

    # Get the total psi = plasma + coils
    psi = eq.psi()

    if show:
        import matplotlib.pyplot as plt
        from .plotting import plotEquilibrium

        if pause > 0.0 and axis is None:
            # No axis specified, so create a new figure
            fig = plt.figure()
            axis = fig.add_subplot(111)

    # Count number of iterations
    iteration = 0

    # Initial relative change in psi (set high to prevent immediate convergence)
    psi_relchange = 10.0

    # Initial psi_bndry (set low to prevent immediate convergence)
    bndry = 0.0

    # Set an initial value for bndry_change (set high to prevent immediate convergence)
    bndry_change = np.inf

    # Plasma assumed to not be limited at first
    has_been_limited = False

    # It is not yet ok to stop itterating
    ok_to_break = False

    psi_maxchange_iterations, psi_relchange_iterations = [], []

    # Start main loop
    while True:
        if show:
            # Plot state of plasma equilibrium
            if pause < 0:
                fig = plt.figure()
                axis = fig.add_subplot(111)
            else:
                axis.clear()

            plotEquilibrium(eq, axis=axis, show=False)

            if pause < 0:
                # Wait for user to close the window
                plt.show()
            else:
                # Update the canvas and pause
                # Note, a short pause is needed to force drawing update
                axis.figure.canvas.draw()
                plt.pause(pause)

        # Copy psi to compare at the end
        psi_last = psi.copy()

        # Boundary flux can also be used as a convergence criterion, so note it
        bndry_last = bndry

        if (iteration >= limit_it or has_been_limited) and check_limited:
            # The user wishes to check for a limited plasma.
            # The minimum number or iterations has passed.
            # If it is ever found to be limited, keep checking for
            # further limited plasmas.

            eq.check_limited = True
            eq.solve(profiles, psi=psi, psi_bndry=eq.psi_bndry)

        else:
            # Either the user does not wish to check for a limited plasma,
            # or not enough iterations have passed yet.
            eq.check_limited = False
            eq.solve(profiles, psi=psi, psi_bndry=psi_bndry)

        # Keep track of whether or not the plasma has at all been limited.
        if eq.is_limited:
            has_been_limited = True

        # If the equilibrium is limited, is must remain so for atleast
        # 1 iteration to allow sudden diverted->limited changes
        # to propagate to the plasma internal profiles. This is captured
        # by also checking if psi_bndry converges.

        if eq.psi_bndry is not None:
            # Check is psi_bndry converges
            bndry = eq.psi_bndry
            bndry_change = bndry_last - bndry
            bndry_relchange = abs(bndry_change / bndry)

        else:
            # Dummy condition to prevent boundary
            # convergence when there is no boundary
            # ie set the change to > rtol
            bndry_relchange = 2.0 * rtol

        # Get the new psi, including coils
        psi = eq.psi()

        # Compare against last solution
        psi_change = psi_last - psi
        psi_maxchange = amax(abs(psi_change))
        psi_relchange = psi_maxchange / (amax(psi) - amin(psi))

        psi_maxchange_iterations.append(psi_maxchange)
        psi_relchange_iterations.append(psi_relchange)

        # User has the option to keep converging until limited
        if not wait_for_limited:
            # User does not wish to wait for the plasma to become limited
            ok_to_break = True

        elif wait_for_limited and eq.is_limited:
            # User wants to check if plasma limited and it is actually limited
            ok_to_break = True

        else:
            # The user wants to wait for a limited plasma. The plasma is not limited.
            ok_to_break = False

        if show:
            print("psi_relchange: " + str(psi_relchange))
            print("bndry_relchange: " + str(bndry_relchange))
            print("bndry_change: " + str(bndry_change))
            print("\n")

        # Check if the changes in psi are small enough and that it is ok to start checking for convergence
        if (
            ((psi_maxchange < atol) or (psi_relchange < rtol))
            and ((bndry_relchange < rtol) or (abs(bndry_change) < atol))
            and ok_to_break
        ):
            break

        # Adjust the coil currents
        if constrain is not None:
            constrain(eq)

        psi = (1.0 - blend) * eq.psi() + blend * psi_last

        # Check if the maximum iterations has been exceeded
        iteration += 1
        if maxits and iteration > maxits:
            raise RuntimeError(
                "Picard iteration failed to converge (too many iterations)"
            )
    if convergenceInfo:
        return array(psi_maxchange_iterations), array(psi_relchange_iterations)
