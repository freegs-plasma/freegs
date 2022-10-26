"""
Plasma control system

Use constraints to adjust coil currents
"""

from numpy import dot, transpose, eye, array, inf
from numpy.linalg import inv, norm
import numpy as np
from scipy import optimize
from . import critical


class constrain(object):
    """
    Adjust coil currents using constraints. To use this class,
    first create an instance by specifying the constraints

    >>> controlsystem = constrain(xpoints = [(1.0, 1.1), (1.0,-1.0)])

    controlsystem will now attempt to create x-points at
    (R,Z) = (1.0, 1.1) and (1.0, -1.1) in any Equilibrium

    >>> controlsystem(eq)

    where eq is an Equilibrium object which is modified by
    the call.

    The constraints which can be set are:

    xpoints - A list of X-point (R,Z) locations

    isoflux - A list of tuples (R1,Z1, R2,Z2)

    psivals - A list of (R,Z,psi) values

    At least one of the above constraints must be included.

    gamma - A scalar, minimises the magnitude of the coil currents

    The following constraitns are entirely optional:

    current_lims - A list of tuples [(l1,u1),(l2,u2)...(lN,uN)] for the upper
    and lower bounds on the currents in each coil.

    max_total_current - The maximum total current through the coilset.
    """

    def __init__(
        self,
        xpoints=[],
        gamma=1e-12,
        isoflux=[],
        psivals=[],
        current_lims=None,
        max_total_current=None,
    ):
        """
        Create an instance, specifying the constraints to apply
        """
        self.xpoints = xpoints
        self.gamma = gamma
        self.isoflux = isoflux
        self.psivals = psivals
        self.current_lims = current_lims
        self.max_total_current = max_total_current

    def __call__(self, eq):
        """
        Apply constraints to Equilibrium eq
        """

        tokamak = eq.getMachine()

        constraint_matrix = []
        constraint_rhs = []
        for xpt in self.xpoints:
            # Each x-point introduces two constraints
            # 1) Br = 0

            Br = eq.Br(xpt[0], xpt[1])

            # Add currents to cancel out this field
            constraint_rhs.append(-Br)
            constraint_matrix.append(tokamak.controlBr(xpt[0], xpt[1]))

            # 2) Bz = 0

            Bz = eq.Bz(xpt[0], xpt[1])

            # Add currents to cancel out this field
            constraint_rhs.append(-Bz)
            constraint_matrix.append(tokamak.controlBz(xpt[0], xpt[1]))

        # Constrain points to have the same flux
        for r1, z1, r2, z2 in self.isoflux:
            # Get Psi at (r1,z1) and (r2,z2)
            p1 = eq.psiRZ(r1, z1)
            p2 = eq.psiRZ(r2, z2)
            constraint_rhs.append(p2 - p1)

            # Coil responses
            c1 = tokamak.controlPsi(r1, z1)
            c2 = tokamak.controlPsi(r2, z2)
            # Control for the difference between p1 and p2
            c = [c1val - c2val for c1val, c2val in zip(c1, c2)]
            constraint_matrix.append(c)

        # Constrain the value of psi
        for r, z, psi in self.psivals:
            p1 = eq.psiRZ(r, z)
            constraint_rhs.append(psi - p1)

            # Coil responses
            c = tokamak.controlPsi(r, z)
            constraint_matrix.append(c)

        if not constraint_rhs:
            raise ValueError("No constraints given")

        # Constraint matrix
        A = array(constraint_matrix)
        b = np.reshape(array(constraint_rhs), (-1,))

        # Number of controls (length of x)
        ncontrols = A.shape[1]

        # First solve analytically by Tikhonov regularisation
        # minimise || Ax - b ||^2 + ||gamma x ||^2

        # Calculate the change in coil current
        self.current_change = dot(
            inv(dot(transpose(A), A) + self.gamma ** 2 * eye(ncontrols)),
            dot(transpose(A), b),
        )

        # Now use the initial analytical soln to guide constrained solve

        # Establish constraints on changes in coil currents from the present
        # and max/min coil current constraints

        current_change_bounds = []

        if self.current_lims is None:
            for i in range(ncontrols):
                current_change_bounds.append((-inf, inf))
        else:
            for i in range(ncontrols):
                cur = tokamak.controlCurrents()[i]
                lower_lim = self.current_lims[i][0] - cur
                upper_lim = self.current_lims[i][1] - cur
                current_change_bounds.append((lower_lim, upper_lim))

        current_change_bnds = array(current_change_bounds)

        # Reform the constraint matrices to include Tikhonov regularisation
        A2 = np.concatenate([A, self.gamma * eye(ncontrols)])
        b2 = np.concatenate([b, np.zeros(ncontrols)])

        # The objetive function to minimize
        # || A2x - b2 ||^2
        def objective(x):
            return (norm((A2 @ x) - b2)) ** 2

        # Additional constraints on the optimisation
        cons = []

        def max_total_currents(x):
            sum = 0.0
            for delta, i in zip(x, tokamak.controlCurrents()):
                sum += abs(delta + i)
            return -(sum - self.max_total_current)

        if self.max_total_current is not None:
            con1 = {"type": "ineq", "fun": max_total_currents}
            cons.append(con1)

        # Use the analytical current change as the initial guess
        if self.current_change.shape[0] > 0:
            x0 = self.current_change
            sol = optimize.minimize(
                objective, x0, method="SLSQP", bounds=current_change_bnds, constraints=cons
            )

            self.current_change = sol.x
            tokamak.controlAdjust(self.current_change)

        # Store info for user
        self.current_change = self.current_change

        # Ensure that the last constraint used is set in the Equilibrium
        eq._constraints = self

    def plot(self, axis=None, show=True):
        """
        Plots constraints used for coil current control

        axis     - Specify the axis on which to plot
        show     - Call matplotlib.pyplot.show() before returning

        """
        from .plotting import plotConstraints

        return plotConstraints(self, axis=axis, show=show)


class ConstrainPsi2D(object):
    """
    Adjusts coil currents to minimise the square differences
    between psi[R,Z] and a target psi.

    Ignores constant offset differences between psi array
    """

    def __init__(self, target_psi, weights=None):
        """
        target_psi : 2D (R,Z) array
            Must be the same size as the equilibrium psi

        weights : float or 2D array of same size as target_psi
            Relative importance of each (R,Z) point in the fitting
            By default every point is equally weighted
            Set points to zero to ignore them in fitting.

        """
        if weights is None:
            weights = np.full(target_psi.shape, 1.0)

        # Remove the average so constant offsets are ignored
        self.target_psi = target_psi - np.average(target_psi, weights=weights)

        self.weights = weights

    def __call__(self, eq):
        """
        Apply constraints to Equilibrium eq
        """

        tokamak = eq.getMachine()
        start_currents = tokamak.controlCurrents()

        end_currents, _ = optimize.leastsq(
            self.psi_difference, start_currents, args=(eq,)
        )

        tokamak.setControlCurrents(end_currents)

        # Ensure that the last constraint used is set in the Equilibrium
        eq._constraints = self

    def psi_difference(self, currents, eq):
        """
        Difference between psi from equilibrium with the given currents
        and the target psi
        """
        eq.getMachine().setControlCurrents(currents)
        psi = eq.psi()
        psi_av = np.average(psi, weights=self.weights)
        return (
            (psi - psi_av - self.target_psi) * self.weights
        ).ravel()  # flatten array


class ConstrainPsiNorm2D(object):
    """
    Adjusts coil currents to minimise the square differences
    between normalised psi[R,Z] and a target normalised psi.
    """

    def __init__(self, target_psinorm, weights=1.0):
        """
        target_psinorm : 2D (R,Z) array
            Must be the same size as the equilibrium psi

        weights : float or 2D array of same size as target_psinorm
            Relative importance of each (R,Z) point in the fitting
            By default every point is equally weighted
            Set points to zero to ignore them in fitting.

        """
        self.target_psinorm = target_psinorm
        self.weights = weights

    def __call__(self, eq):
        """
        Apply constraints to Equilibrium eq
        """

        tokamak = eq.getMachine()
        start_currents = tokamak.controlCurrents()

        end_currents, _ = optimize.leastsq(
            self.psinorm_difference, start_currents, args=(eq,)
        )

        tokamak.setControlCurrents(end_currents)

        # Ensure that the last constraint used is set in the Equilibrium
        eq._constraints = self

    def psinorm_difference(self, currents, eq):
        """
        Difference between normalised psi from equilibrium with the given currents
        and the target psinorm
        """
        eq.getMachine().setControlCurrents(currents)
        psi = eq.psi()

        eq._updateBoundaryPsi(psi)
        psi_bndry = eq.psi_bndry
        psi_axis = eq.psi_axis

        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)

        return (
            (psi_norm - self.target_psinorm) * self.weights
        ).ravel()  # flatten array


class ConstrainPsi2DAdvanced(object):
    """
    Adjusts coil currents to minimise the square differences
    between psi[R,Z] and a target psi.

    Attempts to also constrain the coil currents as in the 'constrain' class.
    """

    def __init__(
        self, target_psi, weights=1.0, current_lims=None, max_total_current=None
    ):
        """
        target_psinorm : 2D (R,Z) array
            Must be the same size as the equilibrium psi

        weights : float or 2D array of same size as target_psinorm
            Relative importance of each (R,Z) point in the fitting
            By default every point is equally weighted
            Set points to zero to ignore them in fitting.

        current_bounds: List of tuples
            Optional list of tuples representing constraints on coil currents to be used
            when reconstructing the equilibrium from the geqdsk file.
            [(l1,u1),(l2,u2)...(lN,uN)]

        Create an instance, specifying the constraints to apply
        """

        self.current_lims = current_lims
        self.target_psi = target_psi
        self.weights = weights

    def __call__(self, eq):
        """
        Apply constraints to Equilibrium eq
        """

        tokamak = eq.getMachine()
        start_currents = np.asarray(tokamak.controlCurrents())
        ncontrols = len(start_currents)

        # In order for the optimisation to work, the initial guess must be within the
        # bounds supplied. Hence, check start_currents and adjust accordingly to be within bounds
        for i in range(ncontrols):
            bnd_upper = max(self.current_lims[i])
            bnd_lower = min(self.current_lims[i])
            sc = start_currents[i]
            if not (bnd_lower <= sc <= bnd_upper):
                if sc < bnd_lower:
                    start_currents[i] = bnd_lower
                else:
                    start_currents[i] = bnd_upper

        current_bounds = []

        for i in range(ncontrols):
            if self.current_lims is None:
                current_bounds.append((-inf, inf))
            else:
                bnd_upper = max(self.current_lims[i])
                bnd_lower = min(self.current_lims[i])
                current_bounds.append((bnd_lower, bnd_upper))

        current_bnds = array(current_bounds)

        # Least squares optimisation of difference in target v achieved normalised psi
        # applied with bounds on coil currents
        end_currents = optimize.minimize(
            self.psi_difference,
            start_currents,
            method="L-BFGS-B",
            bounds=current_bnds,
            args=(eq,),
        ).x

        # Set the latest coil currents
        tokamak.setControlCurrents(end_currents)

        # Ensure that the last constraint used is set in the Equilibrium
        eq._constraints = self

    def psi_difference(self, currents, eq):
        """
        Sum of the squares of the differences between the achieved
        psi and the target psi.
        """
        eq.getMachine().setControlCurrents(currents)
        psi = eq.psi()
        eq._updateBoundaryPsi(psi)

        psi_av = np.average(psi, weights=self.weights)
        diff = (psi - psi_av - self.target_psi) * self.weights
        sum_square_diff = np.sum(diff * diff)

        return sum_square_diff


class ConstrainPsiNorm2DAdvanced(object):
    """
    Adjusts coil currents to minimise the square differences
    between normalised psi[R,Z] and a target normalised psi.

    Attempts to also constrain the coil currents as in the 'constrain' class.
    """

    def __init__(self, target_psinorm, weights=1.0, current_lims=None):
        """
        target_psinorm : 2D (R,Z) array
            Must be the same size as the equilibrium psi

        weights : float or 2D array of same size as target_psinorm
            Relative importance of each (R,Z) point in the fitting
            By default every point is equally weighted
            Set points to zero to ignore them in fitting.

        current_bounds: List of tuples
            Optional list of tuples representing constraints on coil currents to be used
            when reconstructing the equilibrium from the geqdsk file.
            [(l1,u1),(l2,u2)...(lN,uN)]

        Create an instance, specifying the constraints to apply
        """

        self.current_lims = current_lims
        self.target_psinorm = target_psinorm
        self.weights = weights

    def __call__(self, eq):
        """
        Apply constraints to Equilibrium eq
        """

        tokamak = eq.getMachine()
        start_currents = np.asarray(tokamak.controlCurrents())
        ncontrols = len(start_currents)

        # In order for the optimisation to work, the initial guess must be within the
        # bounds supplied. Hence, check start_currents and adjust accordingly to be within bounds
        for i in range(ncontrols):
            bnd_upper = max(self.current_lims[i])
            bnd_lower = min(self.current_lims[i])
            sc = start_currents[i]
            if not (bnd_lower <= sc <= bnd_upper):
                if sc < bnd_lower:
                    start_currents[i] = bnd_lower
                else:
                    start_currents[i] = bnd_upper

        current_bounds = []

        for i in range(ncontrols):
            if self.current_lims is None:
                current_bounds.append((-inf, inf))
            else:
                bnd_upper = max(self.current_lims[i])
                bnd_lower = min(self.current_lims[i])
                current_bounds.append((bnd_lower, bnd_upper))

        current_bnds = array(current_bounds)

        # Least squares optimisation of difference in target v achieved normalised psi
        # applied with bounds on coil currents
        end_currents = optimize.minimize(
            self.psinorm_difference,
            start_currents,
            method="L-BFGS-B",
            bounds=current_bnds,
            args=(eq,),
        ).x

        # Set the latest coil currents
        tokamak.setControlCurrents(end_currents)

        # Ensure that the last constraint used is set in the Equilibrium
        eq._constraints = self

    def psinorm_difference(self, currents, eq):
        """
        Sum of the squares of the differences between the achieved normalised
        psi and the target normalised psi.
        """
        eq.getMachine().setControlCurrents(currents)
        psi = eq.psi()

        eq._updateBoundaryPsi(psi)
        psi_bndry = eq.psi_bndry
        psi_axis = eq.psi_axis

        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)
        diff = (psi_norm - self.target_psinorm) * self.weights
        sum_square_diff = np.sum(diff * diff)

        return sum_square_diff
