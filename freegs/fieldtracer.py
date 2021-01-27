# Field line tracing

from builtins import object

import numpy as np
from scipy.integrate import odeint
from scipy import interpolate

from . import critical


class FieldTracer(object):
    """A class for following magnetic field lines"""

    def __init__(self, eq):
        """
        Initialise a FieldTracer with an Equilibrium eq
        """
        self._eq = eq

        if eq.tokamak.wall:
            # Machine has a wall, used to define edges
            self.updateEvolving = self.wallDomain
        else:
            # No wall, so just use the domain
            self.updateEvolving = self.eqDomain

    def eqDomain(self, R, Z, evolving):
        """Update an array `evolving`, of the same shape as R and Z.
        Set all entries to zero which correspond to (R,Z) points outside
        the domain."""

        eps = 1e-2

        evolving[
            np.logical_or(
                np.logical_or((R < self._eq.Rmin + eps), (R > self._eq.Rmax - eps)),
                np.logical_or((Z < self._eq.Zmin + eps), (Z > self._eq.Zmax - eps)),
            )
        ] = 0.0
        return evolving

    def wallDomain(self, R, Z, evolving):
        """Updates an array `evolving`, of the same shape as R and Z.
        Set all entries to zero which correspond to (R,Z) points outside
        the domain.
        This is done by counting intersections with the wall"""

        Rwall = self._eq.tokamak.wall.R
        Zwall = self._eq.tokamak.wall.Z

        # Location of the middle of the domain
        Rmid = 0.5 * (self._eq.Rmin + self._eq.Rmax)
        Zmid = 0.5 * (self._eq.Zmin + self._eq.Zmax)

        # True if a point is outside the wall
        outside = np.full(R.shape, False)

        nwall = len(Rwall)
        for wi in range(nwall):
            # Find intersection between line from (Rmid, Zmid) to (R,Z)
            # with wall segment from (Rwall[wi], Zwall[wi]) to (Rwall[wi+1], Zwall[wi+1])
            wip = (wi + 1) % nwall

            a = R - Rmid
            b = Rwall[wip] - Rwall[wi]
            c = Z - Zmid
            d = Zwall[wip] - Zwall[wi]

            dr = Rwall[wip] - Rmid
            dz = Zwall[wip] - Zmid

            det = a * d - b * c

            # Note: Here expect divide-by-zero errors occasionally
            # but the resulting values won't be used
            alpha = (d * dr - b * dz) / det  # Location along line 1 [0,1]
            beta = (a * dz - c * dr) / det  # Location along line 2 [0,1]

            # If the lines cross, change outside <-> inside
            # Note: If det is small then lines are nearly parallel
            outside ^= (
                (np.abs(det) > 1e-6)
                & (alpha > 0.0)
                & (alpha < 1.0)
                & (beta > 0.0)
                & (beta < 1.0)
            )
        evolving[outside] = 0.0

    def fieldDirection(self, pos, toroidal_angle, evolving, backward):
        """
        Calculate the magnetic field direction at a given pos
        """
        position = pos.reshape((-1, 3))
        R = position[:, 0]
        Z = position[:, 1]
        # Length is position[:,2]

        # Calculate magnetic field components
        Br = self._eq.Br(R, Z)
        Bz = self._eq.Bz(R, Z)
        Btor = self._eq.Btor(R, Z)

        B = np.sqrt(Br ** 2 + Bz ** 2 + Btor ** 2)

        # Detect when the boundary has been reached
        self.updateEvolving(R, Z, evolving)

        # Common factor in all equations
        evolving_R_Bt = evolving * R / Btor

        # Rate of change of position with toroidal angle phi
        dRdphi = evolving_R_Bt * Br
        dZdphi = evolving_R_Bt * Bz
        dldphi = evolving_R_Bt * B

        if backward:
            # Reverse direction
            dRdphi *= -1.0
            dZdphi *= -1.0

        return np.column_stack((dRdphi, dZdphi, dldphi)).flatten()

    def follow(self, Rstart, Zstart, angles, rtol=None, backward=False):
        """Follow magnetic field lines from (Rstart, Zstart) locations
        to given toroidal angles.
        If backward is True then the field lines are followed in the
        -B direction"""

        Rstart = np.array(Rstart)
        Zstart = np.array(Zstart)

        array_shape = Rstart.shape
        assert Zstart.shape == array_shape

        evolving = np.ones(array_shape)

        # (R,Z,length) with length=0 initially
        position = np.column_stack((Rstart, Zstart, np.zeros(array_shape))).flatten()

        result = odeint(
            self.fieldDirection, position, angles, args=(evolving, backward), rtol=rtol
        )

        return result.reshape(angles.shape + array_shape + (3,))


class LineCoordinates:
    """Coordinates of a field line
    R   Major radius [m]
    Z   Height [m]
    length   Field line length [m]

    All {R, Z, length} are NumPy arrays of the same shape
    """

    def __init__(self, R, Z, length):
        self.R = R
        self.Z = Z
        self.length = length


def traceFieldLines(eq, solwidth=0.03, nlines=10, nturns=50, npoints=200, axis=None):
    """Trace field lines from the outboard midplane

    Inputs
    ------

    eq          Equilibrium object
    solwidth    The width of the SOL in meters
    nlines      Number of field lines to follow
    nturns      Number of times around the tokamak to follow
    npoints     Maximum number of points per line. May hit a wall

    axis        Matplotlib figure Axis. If given, field lines
                are plotted on the axis

    Returns
    -------

    The forward and backward field line coordinates
    stored in LineCoordinates objects

    >>> forward, backward = traceFieldLines(eq)

    forward and backward have data members
    R, Z, length  2D arrays of shape (npoints, nlines)

    """
    ft = FieldTracer(eq)

    # Find the edge of the plasma
    psi = eq.psi()
    opoint, xpoint = critical.find_critical(eq.R, eq.Z, psi)

    r0, z0, psi_axis = opoint[0]
    psi_bndry = xpoint[0][2]

    # Find outboard midplane
    psifunc = interpolate.RectBivariateSpline(
        eq.R[:, 0], eq.Z[0, :], (psi - psi_axis) / (psi_bndry - psi_axis)
    )

    rmid, zmid = critical.find_psisurface(
        eq, psifunc, r0, z0, r0 + 10.0, z0, psival=1.0
    )

    # Starting locations, just outside the separatrix
    # Start first point a bit away from separatrix
    rstart = np.linspace(rmid + (solwidth / nlines), rmid + solwidth, nlines)
    zstart = np.full(nlines, zmid)

    angles = np.linspace(0.0, nturns * 2 * np.pi, npoints)

    # Follow field lines
    line_forward = ft.follow(rstart, zstart, angles)
    line_backward = ft.follow(rstart, zstart, angles, backward=True)

    forward = LineCoordinates(
        line_forward[:, :, 0], line_forward[:, :, 1], line_forward[:, :, 2]
    )

    backward = LineCoordinates(
        line_backward[:, :, 0], line_backward[:, :, 1], line_backward[:, :, 2]
    )

    if axis:
        # Plot field lines
        axis.plot(forward.R, forward.Z)
        axis.plot(backward.R, backward.Z)

        # Mark the starting location
        axis.plot(rstart, zstart, color="k", linewidth=2)

    return forward, backward
