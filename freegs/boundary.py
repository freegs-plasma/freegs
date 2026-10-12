"""
Functions to impose boundary conditions on psi(plasma)

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

from numpy import (
    arange,
    concatenate,
    full,
    log,
    ones_like,
    pi,
    prod,
    sqrt,
    stack,
    where,
    zeros,
)
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.polynomial import polyfromroots
from scipy.integrate import romb  # Romberg integration
from scipy.special import ellipe, ellipk, ellipkm1

from .gradshafranov import Greens, mu0


def fixedBoundary(eq, Jtor, psi):
    """
    Set psi=0 on all boundaries

    Inputs
    ------

    eq    Equilibrium object (not used)
    Jtor  2D array of toroidal current (not used)
    psi   2D array of psi values (modified by call)

    Returns
    -------
    None
    """
    psi[0, :] = 0.0
    psi[:, 0] = 0.0
    psi[-1, :] = 0.0
    psi[:, -1] = 0.0


def freeBoundary(eq, Jtor, psi):
    """
    Apply a free boundary condition using Green's functions

    Note: This method is inefficient because it requires
    an integral over the area of the domain for each point
    on the boundary.

    Inputs
    ------

    eq    Equilibrium object (not used)
    Jtor  2D array of toroidal current (not used)
    psi   2D array of psi values (modified by call)

    Returns
    -------

    None
    """

    # Get the (R,Z) coordinates from the Equilibrium object
    R = eq.R
    Z = eq.Z

    nx, ny = psi.shape

    dR = R[1, 0] - R[0, 0]
    dZ = Z[0, 1] - Z[0, 0]

    # List of indices on the boundary
    bndry_indices = concatenate(
        [
            [(x, 0) for x in range(nx)],
            [(x, ny - 1) for x in range(nx)],
            [(0, y) for y in range(ny)],
            [(nx - 1, y) for y in range(ny)],
        ]
    )

    for x, y in bndry_indices:
        # Calculate the response of the boundary point
        # to each cell in the plasma domain
        greenfunc = Greens(R, Z, R[x, y], Z[x, y])

        # Prevent infinity/nan by removing (x,y) point
        greenfunc[x, y] = 0.0

        # Integrate over the domain
        psi[x, y] = romb(romb(greenfunc * Jtor)) * dR * dZ


def freeBoundaryHagenow(eq, Jtor, psi):
    """
    Apply a free boundary using von Hagenow's method

    Inputs
    ------

    eq    Equilibrium object (not used)
    Jtor  2D array of toroidal current (not used)
    psi   2D array of psi values (modified by call)

    Returns
    -------

    None
    """

    # Get the (R,Z) coordinates from the Equilibrium object
    R = eq.R
    Z = eq.Z

    nx, ny = psi.shape

    dR = R[1, 0] - R[0, 0]
    dZ = Z[0, 1] - Z[0, 0]

    # solver RHS
    rhs = eq.R * Jtor

    # Apply a zero-value boundary condition
    rhs[0, :] = 0.0
    rhs[:, 0] = 0.0
    rhs[-1, :] = 0.0
    rhs[:, -1] = 0.0

    # Solve with a fixed boundary
    psi_fixed = eq.callSolver(psi, rhs)

    # Differentiate at the boundary
    # Note: normal is out of the domain, so this is -dU/dx

    # Fourth-order one-sided differences
    coeffs = [(0, 25.0 / 12), (1, -4.0), (2, 3.0), (3, -16.0 / 12), (4, 1.0 / 4)]

    dUdn_L = (
        sum([weight * psi_fixed[index, :] for index, weight in coeffs]) / dR
    )  # left boundary

    dUdn_R = (
        sum([weight * psi_fixed[-(1 + index), :] for index, weight in coeffs]) / dR
    )  # Right boundary

    dUdn_D = (
        sum([weight * psi_fixed[:, index] for index, weight in coeffs]) / dZ
    )  # Down boundary

    dUdn_U = (
        sum([weight * psi_fixed[:, -(1 + index)] for index, weight in coeffs]) / dZ
    )  # Upper boundary

    dd = sqrt(dR**2 + dZ**2)  # Diagonal spacing

    # Left down corner
    dUdn_L[0] = dUdn_D[0] = (
        sum([weight * psi_fixed[index, index] for index, weight in coeffs]) / dd
    )

    # Left upper corner
    dUdn_L[-1] = dUdn_U[0] = (
        sum([weight * psi_fixed[index, -(1 + index)] for index, weight in coeffs]) / dd
    )

    # Right down corner
    dUdn_R[0] = dUdn_D[-1] = (
        sum([weight * psi_fixed[-(1 + index), index] for index, weight in coeffs]) / dd
    )

    # Right upper corner
    dUdn_R[-1] = dUdn_U[-1] = (
        sum(
            [weight * psi_fixed[-(1 + index), -(1 + index)] for index, weight in coeffs]
        )
        / dd
    )

    # Now for each point on the boundary perform a loop integral
    #
    # The observation points are the boundary grid points themselves. The
    # kernel has a logarithmic singularity where the source coincides with
    # the observation point; this is integrated by _edge_matrix.

    # Boundary grid points, as (x index, y index)
    xs = concatenate(
        [arange(nx), arange(nx), zeros(ny - 2, dtype=int), full(ny - 2, nx - 1)]
    )
    ys = concatenate(
        [zeros(nx, dtype=int), full(nx, ny - 1), arange(1, ny - 1), arange(1, ny - 1)]
    )

    # The weights depend only on the grid, so are kept for later calls
    key = (R[0, 0], R[-1, 0], Z[0, 0], Z[0, -1], nx, ny)
    cache = getattr(eq, "_hagenow_weights", None)
    if cache is None or cache[0] != key:
        Robs = R[xs, ys]
        Zobs = Z[xs, ys]

        # Each edge: R, Z of the nodes, node spacing, and the index of
        # each observation point along the edge (or -1 if it is not on it)
        edges = [
            (R[0, :], Z[0, :], dZ, where(xs == 0, ys, -1)),  # Left
            (R[-1, :], Z[-1, :], dZ, where(xs == nx - 1, ys, -1)),  # Right
            (R[:, 0], Z[:, 0], dR, where(ys == 0, xs, -1)),  # Down
            (R[:, -1], Z[:, -1], dR, where(ys == ny - 1, xs, -1)),  # Upper
        ]
        # The integrand is G * (dU/dn) / R
        weights = [
            _edge_matrix(Redge, Zedge, step, Robs, Zobs, index) / Redge
            for Redge, Zedge, step, index in edges
        ]
        eq._hagenow_weights = cache = (key, weights)

    psi[xs, ys] = sum(
        matrix @ dUdn
        for matrix, dUdn in zip(cache[1], [dUdn_L, dUdn_R, dUdn_D, dUdn_U])
    )


def _greens_split(Rs, Zs, Ro, Zo, singular=False):
    """
    Green's function G(Rs, Zs -> Ro, Zo), as in gradshafranov.Greens but
    accurate close to the source, and the coefficient B of ln(1/m1) in it,
    where m1 = 1 - k^2 = ((Rs-Ro)^2 + (Zs-Zo)^2) / ((Rs+Ro)^2 + (Zs-Zo)^2)

    The expansion of the complete elliptic integrals about k = 1 is
    K(k^2) = K(m1)/pi ln(1/m1) + ..., E(k^2) = (K(m1)-E(m1))/pi ln(1/m1) + ...

    Returns (G, B, d2); G is not evaluated (None) if singular is True
    since it is infinite at the source.
    """
    d2 = (Rs - Ro) ** 2 + (Zs - Zo) ** 2
    m1 = d2 / ((Rs + Ro) ** 2 + (Zs - Zo) ** 2)
    k2 = 1.0 - m1
    pref = (mu0 / (2.0 * pi)) * sqrt(Rs * Ro / k2)

    K1 = ellipk(m1)
    E1 = ellipe(m1)
    B = pref * ((2.0 - k2) * K1 - 2.0 * (K1 - E1)) / pi

    G = None
    if not singular:
        G = pref * ((2.0 - k2) * ellipkm1(m1) - 2.0 * ellipe(k2))
    return G, B, d2


def _lagrange(nodes, x):
    """
    Lagrange basis polynomials through the nodes, evaluated at the array x.
    Returns an array of shape x.shape + (len(nodes),)
    """
    basis = []
    for a in nodes:
        value = ones_like(x)
        for b in nodes:
            if b != a:
                value = value * (x - b) / (a - b)
        basis.append(value)
    return stack(basis, axis=-1)


def _log_panel_weights(taus, offset, step):
    """
    Weights w[a] = integral over a panel of ln(step*|t - tau|) l_a(t) dt,
    where t in [0,1] is the position in the panel, tau an integer position
    of the singularity relative to the panel start (an array), and l_a
    the cubic Lagrange polynomial on the nodes 0..3 of the stencil, for a panel
    starting at node `offset` of the stencil. Returns an array (len(taus), 4).
    """
    x, w = leggauss(10)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    basis = _lagrange(range(4), offset + x)  # (q, 4)

    # Smooth case: Gauss-Legendre
    distance = abs(x[None, :] - taus[:, None])
    result = (w * log(step * distance)) @ basis

    # Singularity at the start (tau = 0) or end (tau = 1) of the panel:
    # integrals of t^k ln t and t^k ln(1-t), exact
    for a in range(4):
        roots = [n for n in range(4) if n != a]
        norm = prod([a - n for n in roots])
        # Polynomial in t of l_a(offset + t)
        poly = polyfromroots([n - offset for n in roots]) / norm
        weights = poly * log(step) / (arange(len(poly)) + 1.0)
        smooth = sum(weights)  # integral of ln(step) * l_a
        for tau, moment in (
            (0, lambda k: -1.0 / (k + 1) ** 2),
            (1, lambda k: -sum(1.0 / j for j in range(1, k + 2)) / (k + 1)),
        ):
            value = smooth + sum(poly[k] * moment(k) for k in range(len(poly)))
            result[taus == tau, a] = value
    return result


def _edge_matrix(Redge, Zedge, step, Robs, Zobs, index):
    """
    Quadrature weights for the integral over a straight, uniformly spaced
    boundary edge of the Green's function G(R, Z; Robs, Zobs) times a smooth
    function f given at the nodes of the edge.

    f is interpolated by piecewise cubic polynomials, and the product with
    the kernel is integrated over each panel, so the result is fourth order
    accurate. Where an observation point lies on the edge the kernel has a
    logarithmic singularity, G = H - 2 B ln|s - s0| with H and B smooth.
    B comes from the logarithmic terms of the complete elliptic integrals;
    the term with B is integrated analytically against the interpolating
    polynomials, and the smooth remainder H by Gauss-Legendre quadrature.

    Redge, Zedge   Coordinates of the edge nodes (n,)
    step           Node spacing
    Robs, Zobs     Observation points (m,)
    index          Index of each observation point along this edge, or -1
                   if it is not on the edge (m,)

    Returns an array of shape (m, n) of weights.
    """
    n = len(Redge)
    A = zeros((len(Robs), n))

    x, w = leggauss(8)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w

    on = index >= 0
    sel = where(on)[0]

    for p in range(n - 1):
        start = min(max(p - 1, 0), n - 4)
        basis = _lagrange(range(4), p - start + x)  # (q, 4)

        Rq = Redge[p] + x * (Redge[p + 1] - Redge[p])
        Zq = Zedge[p] + x * (Zedge[p + 1] - Zedge[p])

        G, B, d2 = _greens_split(Rq, Zq, Robs[:, None], Zobs[:, None])
        # For points on the edge, remove the logarithmic part
        # B ln(1/d2) is integrated separately below
        G[sel] += B[sel] * log(d2[sel])
        A[:, start : start + 4] += step * (G * w) @ basis

    if len(sel) > 0:
        # Logarithmic part: -2 B(s) ln|s - s0| f(s), interpolating B f
        _, B, _ = _greens_split(
            Redge[None, :],
            Zedge[None, :],
            Robs[sel, None],
            Zobs[sel, None],
            singular=True,
        )
        # W[i, j]: integral of ln|s - s_i| l_j(s) ds
        W = zeros((n, n))
        nodes = arange(n)
        for p in range(n - 1):
            start = min(max(p - 1, 0), n - 4)
            W[:, start : start + 4] += step * _log_panel_weights(
                (nodes - p).astype(float), p - start, step
            )
        A[sel] += -2.0 * B * W[index[sel]]
    return A
