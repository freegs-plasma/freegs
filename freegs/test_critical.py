import numpy as np

from . import critical


def test_one_opoint():
    nx = 65
    ny = 65

    r1d = np.linspace(1.0, 2.0, nx)
    z1d = np.linspace(-1.0, 1.0, nx)
    r2d, z2d = np.meshgrid(r1d, z1d, indexing="ij")

    r0 = 1.5
    z0 = 0.0

    # This has one O-point at (r0,z0) and no x-points
    def psi_func(R, Z):
        return np.exp(-((R - r0) ** 2 + (Z - z0) ** 2) / 0.3**2)

    opoints, xpoints = critical.find_critical(r2d, z2d, psi_func(r2d, z2d))

    assert len(xpoints) == 0
    assert len(opoints) == 1
    assert np.isclose(opoints[0][0], r0, atol=1.0 / nx)
    assert np.isclose(opoints[0][1], z0, atol=1.0 / ny)


def test_one_xpoint():
    nx = 65
    ny = 65

    r1d = np.linspace(1.0, 2.0, nx)
    z1d = np.linspace(-1.0, 1.0, nx)
    r2d, z2d = np.meshgrid(r1d, z1d, indexing="ij")

    r0 = 1.5
    z0 = 0.0

    # This has one X-point at (r0,z0) and no O-points
    def psi_func(R, Z):
        return (R - r0) ** 2 - (Z - z0) ** 2

    opoints, xpoints = critical.find_critical(r2d, z2d, psi_func(r2d, z2d))

    assert len(xpoints) == 1
    assert len(opoints) == 0
    assert np.isclose(xpoints[0][0], r0, atol=1.0 / nx)
    assert np.isclose(xpoints[0][1], z0, atol=1.0 / ny)


def test_doublet():
    nx = 65
    ny = 65

    r1d = np.linspace(1.0, 2.0, nx)
    z1d = np.linspace(-1.0, 1.0, nx)
    r2d, z2d = np.meshgrid(r1d, z1d, indexing="ij")

    r0 = 1.5
    z0 = 0.1

    # This has two O-points, and one x-point at (r0, z0)
    def psi_func(R, Z):
        return np.exp(-((R - r0) ** 2 + (Z - z0 - 0.3) ** 2) / 0.3**2) + np.exp(
            -((R - r0) ** 2 + (Z - z0 + 0.3) ** 2) / 0.3**2
        )

    opoints, xpoints = critical.find_critical(r2d, z2d, psi_func(r2d, z2d))

    assert len(xpoints) == 1
    assert len(opoints) == 2
    assert np.isclose(xpoints[0][0], r0, atol=1.0 / nx)
    assert np.isclose(xpoints[0][1], z0, atol=1.0 / ny)


def test_mask_zero_psi_bndry():
    nx = 65
    ny = 65

    r1d = np.linspace(1.0, 2.0, nx)
    z1d = np.linspace(-1.0, 1.0, nx)
    r2d, z2d = np.meshgrid(r1d, z1d, indexing="ij")

    r0 = 1.5
    z0 = 0.0

    # This has one O-point at (r0,z0) and no x-points
    # Range from around -0.5 to +0.5
    def psi_func(R, Z):
        return np.exp(-((R - r0) ** 2 + (Z - z0) ** 2) / 0.3**2) - 0.5

    psi = psi_func(r2d, z2d)

    opoints, xpoints = critical.find_critical(r2d, z2d, psi)

    assert len(xpoints) == 0
    assert len(opoints) == 1
    assert np.isclose(opoints[0][0], r0, atol=1.0 / nx)
    assert np.isclose(opoints[0][1], z0, atol=1.0 / ny)

    mask = critical.core_mask(r2d, z2d, psi, opoints, xpoints, psi_bndry=0.0)

    # Some of the mask must equal 1, some 0
    assert np.any(np.equal(mask, 1))
    assert np.any(np.equal(mask, 0))
