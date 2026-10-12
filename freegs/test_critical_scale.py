import numpy as np
import pytest
from scipy import interpolate

from . import critical


def _grid(nx):
    r1d = np.linspace(1.0, 2.0, nx)
    z1d = np.linspace(-1.0, 1.0, nx)
    return np.meshgrid(r1d, z1d, indexing="ij")


def _two_o_one_x(R, Z, r0=1.5, z0=0.0, a=0.3, w=0.3):
    """Two Gaussian wells at (r0, z0 +- a) with a saddle at (r0, z0) by symmetry"""
    return np.exp(-((R - r0) ** 2 + (Z - z0 - a) ** 2) / w**2) + np.exp(
        -((R - r0) ** 2 + (Z - z0 + a) ** 2) / w**2
    )


@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 1e3, -1e-3, -1.0])
def test_critical_invariant_to_psi_scaling(scale):
    """The critical points depend on the shape of psi, not on its units"""
    R, Z = _grid(65)
    psi = _two_o_one_x(R, Z)
    ref_o, ref_x = critical.find_critical(R, Z, psi)
    opoints, xpoints = critical.find_critical(R, Z, scale * psi)

    assert len(ref_o) == 2
    assert len(ref_x) == 1
    assert len(opoints) == 2
    assert len(xpoints) == 1

    # The saddle is at the symmetry point, whatever the scale
    assert xpoints[0][0] == pytest.approx(1.5, abs=1e-6)
    assert xpoints[0][1] == pytest.approx(0.0, abs=1e-6)
    for p, q in zip(
        sorted(opoints, key=lambda o: o[1]), sorted(ref_o, key=lambda o: o[1])
    ):
        assert p[0] == pytest.approx(q[0], abs=1e-6)
        assert p[1] == pytest.approx(q[1], abs=1e-6)
        assert p[2] == pytest.approx(scale * q[2], rel=1e-6)


def test_critical_points_are_converged():
    """The gradient at the returned points is small compared to the curvature"""
    R, Z = _grid(65)
    psi = 1e-3 * _two_o_one_x(R, Z)
    opoints, xpoints = critical.find_critical(R, Z, psi)
    f = interpolate.RectBivariateSpline(R[:, 0], Z[0, :], psi)
    assert len(opoints) + len(xpoints) == 3
    for r, z, _ in opoints + xpoints:
        grad = np.hypot(f(r, z, dx=1)[0, 0], f(r, z, dy=1)[0, 0])
        hess = abs(f(r, z, dx=2)[0, 0]) + abs(f(r, z, dy=2)[0, 0])
        assert grad < 1e-6 * (2.0 / 64) * hess
