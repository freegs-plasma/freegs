import numpy as np
import pytest

from . import control, equilibrium, jtor, picard
from .critical import find_critical
from .machine import TestTokamak


class FixedPointEquilibrium:
    """
    Minimal stand-in for an Equilibrium whose Picard map is known exactly:
    psi -> psi_ref + mapping(psi - psi_ref), with the unique fixed point psi_ref
    and psi_bndry = psi at the first point, reported as the offset from psi_ref.
    """

    is_limited = False
    check_limited = False

    def __init__(self, mapping, offset, bndry=None):
        self.psi_ref = np.linspace(0.0, 1.0, 5)
        self.mapping = mapping
        self.plasma_psi = self.psi_ref + offset
        self.psi_bndry = bndry

    def psi(self):
        return self.plasma_psi

    def solve(self, profiles, psi=None, psi_bndry=None):
        d = psi - self.psi_ref
        self._updatePlasmaPsi(self.psi_ref + self.mapping(d))

    def _updatePlasmaPsi(self, plasma_psi):
        self.plasma_psi = plasma_psi
        if self.psi_bndry is not None:
            self.psi_bndry = float((plasma_psi - self.psi_ref)[0])


def cycling_map(d):
    # Slope -3 at the fixed point d=0 (unstable) saturating at +-a: the plain
    # iteration from d=2a is trapped in the exact two-cycle d = +-3a
    a = 0.1
    return -3.0 * np.clip(d, -a, a)


def test_picard_two_cycle_plain_fails():
    eq = FixedPointEquilibrium(cycling_map, 0.2, bndry=0.0)
    with pytest.raises(RuntimeError):
        picard.solve(eq, None, rtol=1e-8, maxits=60, stabilise=False)


def test_picard_two_cycle_stabilised_converges_to_fixed_point():
    eq = FixedPointEquilibrium(cycling_map, 0.2, bndry=0.0)
    picard.solve(eq, None, rtol=1e-8, maxits=60)
    # The unique fixed point of the map is psi_ref, where psi_bndry = 0
    assert np.allclose(eq.psi(), eq.psi_ref, atol=1e-7)
    assert abs(eq.psi_bndry) < 1e-7


def test_picard_converging_iteration_unchanged():
    # A converging iteration is not altered by the stabilisation
    def contraction(d):
        return -0.5 * d

    results = []
    for stabilise in (False, True):
        eq = FixedPointEquilibrium(contraction, 0.3, bndry=0.0)
        its = picard.solve(
            eq, None, rtol=1e-9, maxits=100, stabilise=stabilise, convergenceInfo=True
        )
        results.append((len(its[0]), eq.psi().copy()))
    assert results[0][0] == results[1][0]
    assert np.array_equal(results[0][1], results[1][1])
    assert np.allclose(results[0][1], FixedPointEquilibrium(None, 0).psi_ref, atol=1e-8)


def test_picard_zero_boundary_flux():
    # psi_bndry == 0 for the whole iteration must not divide by zero
    eq = FixedPointEquilibrium(lambda d: 0.5 * d, 0.3)
    eq.psi_bndry = 0.0
    eq._updatePlasmaPsi = lambda p: setattr(eq, "plasma_psi", p)
    picard.solve(eq, None, rtol=1e-6, maxits=100)
    assert np.allclose(eq.psi(), eq.psi_ref, atol=1e-5)


def _solve_near_double_null(asymmetry, initial_psi=None, rtol=1e-6):
    eq = equilibrium.Equilibrium(
        tokamak=TestTokamak(), Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0, nx=33, ny=33
    )
    profiles = jtor.ConstrainPaxisIp(eq, 1e3, 2e5, 2.0)
    # Two X-points on nearly the same flux surface
    constrain = control.constrain(xpoints=[(1.1, -0.8), (1.1, 0.8 - asymmetry)])
    if initial_psi is not None:
        eq._updatePlasmaPsi(initial_psi)
    picard.solve(eq, profiles, constrain, rtol=rtol, maxits=150)
    return eq


def test_near_double_null_unique_fixed_point():
    eq1 = _solve_near_double_null(0.01)

    # The X-point fluxes are close, so the primary X-point is nearly ambiguous
    opt, xpt = find_critical(eq1.R, eq1.Z, eq1.psi())
    assert len(xpt) == 2
    flux_span = eq1.psi_axis - eq1.psi_bndry
    assert abs(xpt[0][2] - xpt[1][2]) < 0.01 * abs(flux_span)

    # A different initial guess reaches the same fixed point
    eq2 = _solve_near_double_null(0.01, initial_psi=0.5 * eq1.plasma_psi)
    scale = np.max(np.abs(eq1.psi()))
    assert np.max(np.abs(eq1.psi() - eq2.psi())) < 1e-3 * scale
    assert eq1.psi_bndry == pytest.approx(eq2.psi_bndry, rel=1e-3)

    # Self-consistency: one more Picard step leaves the solution unchanged
    profiles = jtor.ConstrainPaxisIp(eq1, 1e3, 2e5, 2.0)
    psi = eq1.psi().copy()
    eq1.solve(profiles, psi=psi)
    assert np.max(np.abs(eq1.psi() - psi)) < 1e-3 * scale
