"""Independent integral oracle for signed constant-q flux profiles."""
import numpy as np
import pytest

from .equilibrium import Equilibrium


@pytest.mark.parametrize("edge", [-2.0, 2.0])
@pytest.mark.parametrize("qvalue", [-1.5, 1.5])
def test_toroidal_flux_uses_axis_reference_for_every_request(edge, qvalue):
    eq = object.__new__(Equilibrium)
    eq.psi_axis, eq.psi_bndry = 0.0, edge
    eq.q = lambda s: np.full(np.asarray(s).shape, qvalue)
    points = edge * np.array([0.7, 0.2, 1.0])
    expected = 2 * np.pi * qvalue * points
    np.testing.assert_allclose(eq.tor_flux(points), expected)
    assert eq.tor_flux(points[0]) == pytest.approx(expected[0])
    profile = eq.tor_flux()
    np.testing.assert_allclose(profile, 2 * np.pi * qvalue * np.linspace(0, edge, 101))
    np.testing.assert_allclose(eq.rhotor(points), np.sqrt(points / edge))
    assert eq.rhotor(0.0) == pytest.approx(0.0)
    assert eq.rhotor(edge) == pytest.approx(1.0)
