"""
Tests of the coil current control system
"""

import numpy as np
from scipy.optimize import lsq_linear

from . import control, machine
from .equilibrium import Equilibrium


def _machine_and_targets():
    # A series pair of coils (one control) and four independent coils
    cs = machine.Circuit(
        [
            ("U", machine.Coil(0.5, 0.4), 1.0),
            ("L", machine.Coil(0.5, -0.4), 1.0),
        ]
    )
    tokamak = machine.Machine(
        [
            ("CS", cs),
            ("P1", machine.Coil(1.8, 0.8)),
            ("P2", machine.Coil(1.8, -0.8)),
            ("P3", machine.Coil(1.2, 1.1)),
            ("P4", machine.Coil(1.2, -1.1)),
        ]
    )
    eq = Equilibrium(
        tokamak=tokamak, Rmin=0.1, Rmax=2.0, Zmin=-1.2, Zmax=1.2, nx=33, ny=33
    )
    points = [
        (1.0, 0.0),
        (1.2, 0.3),
        (1.0, -0.3),
        (1.4, 0.1),
        (0.9, 0.2),
        (1.3, -0.2),
        (1.1, 0.05),
    ]
    values = [0.02, 0.015, 0.02, 0.01, 0.02, 0.012, 0.017]
    return eq, tokamak, points, values


def test_constrain_bounded_matches_lsq_linear():
    """
    With binding current limits the result must be the bounded least squares
    optimum, computed here independently with scipy.optimize.lsq_linear.
    """
    eq, tokamak, points, values = _machine_and_targets()

    A = np.array([tokamak.controlPsi(r, z) for r, z in points])
    b = np.array(values) - np.array([eq.psiRZ(r, z) for r, z in points])

    # Make every limit bind
    free = np.linalg.lstsq(A, b, rcond=None)[0]
    limit = 0.5 * np.abs(free)
    ref = lsq_linear(A, b, bounds=(-limit, limit), tol=1e-14)

    control.constrain(
        psivals=[(r, z, v) for (r, z), v in zip(points, values)],
        current_lims=[(-lim, lim) for lim in limit],
        gamma=0.0,
    )(eq)

    currents = np.array(tokamak.controlCurrents())
    assert np.all(np.abs(currents) <= limit * (1 + 1e-9))

    cost = 0.5 * np.sum((A @ currents - b) ** 2)
    assert cost <= ref.cost * (1 + 1e-6)


def test_constrain_total_current_limit():
    """The total current limit is still enforced"""
    eq, tokamak, points, values = _machine_and_targets()

    A = np.array([tokamak.controlPsi(r, z) for r, z in points])
    b = np.array(values) - np.array([eq.psiRZ(r, z) for r, z in points])
    free = np.linalg.lstsq(A, b, rcond=None)[0]

    max_total = 0.5 * np.sum(np.abs(free))
    control.constrain(
        psivals=[(r, z, v) for (r, z), v in zip(points, values)],
        max_total_current=max_total,
    )(eq)

    assert np.sum(np.abs(tokamak.controlCurrents())) <= max_total * (1 + 1e-6)
