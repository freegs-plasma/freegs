import numpy as np
import pytest

from . import critical


def test_primary_opoint_not_nearest_to_centre():
    """
    A deep well near the edge of the domain and a shallow vacuum O-point at the
    centre of the domain. The primary O-point is the well, which has an X-point
    connected to it, not the O-point nearest to the centre.
    """
    r1d = np.linspace(1.0, 3.0, 129)
    z1d = np.linspace(-1.0, 1.0, 129)
    R, Z = np.meshgrid(r1d, z1d, indexing="ij")
    k = 3.0
    psi = np.exp(-((R - 1.3) ** 2 + Z**2) / 0.06) + 0.05 * np.cos(
        k * (R - 2.0)
    ) * np.cos(k * Z)

    opoints, xpoints = critical.find_critical(R, Z, psi)

    assert len(opoints) >= 2
    i, j = np.unravel_index(np.argmax(psi), psi.shape)
    assert opoints[0][2] == pytest.approx(psi.max(), rel=1e-3)
    assert opoints[0][0] == pytest.approx(R[i, j], abs=0.02)
    assert opoints[0][1] == pytest.approx(Z[i, j], abs=0.02)
    # The saddle between the two O-points is connected to the primary one
    assert any(abs(x[0] - 1.9) < 0.1 and abs(x[1]) < 0.01 for x in xpoints)


def test_primary_opoint_ignores_grid_scale_spikes():
    """
    A resolved well at the centre of the domain and a lattice of grid-scale
    spikes, like the filaments of a coil inside the grid. The spikes are
    O-points connected to many saddles, but are not the magnetic axis.
    """
    n = 257
    r1d = np.linspace(1.0, 3.0, n)
    z1d = np.linspace(-1.0, 1.0, n)
    R, Z = np.meshgrid(r1d, z1d, indexing="ij")
    h = r1d[1] - r1d[0]
    psi = 1.0 * np.exp(-((R - 2.0) ** 2 + Z**2) / 0.08)
    for rc in (2.6, 2.7, 2.8):
        for zc in (0.5, 0.6, 0.7):
            psi += 1.2 * np.exp(-((R - rc) ** 2 + (Z - zc) ** 2) / (2.0 * h) ** 2)

    opoints, xpoints = critical.find_critical(R, Z, psi)

    assert len(opoints) > 2
    assert opoints[0][0] == pytest.approx(2.0, abs=0.01)
    assert opoints[0][1] == pytest.approx(0.0, abs=0.01)
