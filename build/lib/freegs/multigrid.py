"""
Multigrid solver for elliptic problems

Example
-------

$ python multigrid.py

This will run the test case, solving Poisson equation in 2D

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

from numpy import zeros, max, abs, reshape
from scipy.sparse.linalg import factorized
from scipy.sparse import eye


class MGDirect:
    def __init__(self, A):
        self.solve = factorized(A.tocsc())  # LU decompose

    def __call__(self, x, b):
        b1d = reshape(b, -1)  # 1D view

        x = self.solve(b1d)

        return reshape(x, b.shape)


class MGJacobi:
    def __init__(self, A, ncycle=4, niter=10, subsolver=None):
        """
        Initialise solver

        A   - The matrix to solve
        subsolver - An operator at lower resolution
        ncycle - Number of V-cycles
        niter - Number of Jacobi iterations

        """
        self.A = A
        self.diag = A.diagonal()
        self.subsolver = subsolver
        self.niter = niter
        self.ncycle = ncycle

        self.sub_b = None
        self.xupdate = None

    def __call__(self, xi, bi, ncycle=None, niter=None):
        """
        Solve Ax = b, given initial guess for x

        ncycle - Optional number of cycles

        """

        # Need to reshape x and b into 1D arrays
        x = reshape(xi, -1)
        b = reshape(bi, -1)

        if ncycle is None:
            ncycle = self.ncycle
        if niter is None:
            niter = self.niter

        for c in range(ncycle):
            # Jacobi smoothing
            for i in range(niter):
                x += (b - self.A.dot(x)) / self.diag

            if self.subsolver:
                # Calculate the error
                error = b - self.A.dot(x)

                # Restrict error onto coarser mesh
                self.sub_b = restrict(reshape(error, xi.shape))

                # smooth this error
                sub_x = zeros(self.sub_b.shape)
                sub_x = self.subsolver(sub_x, self.sub_b)

                # Prolong the solution
                self.xupdate = interpolate(sub_x)

                x += reshape(self.xupdate, -1)

            # Jacobi smoothing
            for i in range(niter):
                x += (b - self.A.dot(x)) / self.diag

        return x.reshape(xi.shape)


def createVcycle(nx, ny, generator, nlevels=4, ncycle=1, niter=10, direct=True):
    """
    Create a hierarchy of solvers in a multigrid V-cycle

    nx, ny - The highest resolution
    generator(nx,ny) - Returns a sparse matrix, given resolution
    nlevels - Number of multigrid levels
    direct - Lowest level uses direct solver
    ncycle - Number of V cycles. This is only passed to the top level MGJacobi object
    niter - Number of Jacobi iterations per level

    """

    if (nx - 1) % 2 == 1 or (ny - 1) % 2 == 1:
        # Can't divide any further
        nlevels = 1

    if nlevels > 1:
        # Create the solver at lower resolution

        nxsub = (nx - 1) // 2 + 1
        nysub = (ny - 1) // 2 + 1

        subsolver = createVcycle(
            nxsub, nysub, generator, nlevels - 1, niter=niter, direct=direct
        )

        # Create the sparse matrix
        A = generator(nx, ny)
        # Create the solver
        return MGJacobi(A, niter=niter, subsolver=subsolver, ncycle=ncycle)

    # At lowest level

    # Create the sparse matrix
    A = generator(nx, ny)
    if direct:
        return MGDirect(A)
    return MGJacobi(A, niter=niter, ncycle=ncycle, subsolver=None)


def smoothJacobi(A, x, b, dx, dy):
    """
    Smooth the solution using Jacobi method
    """

    if b.shape != x.shape:
        raise ValueError("b and x have different shapes")

    smooth = x + (b - A(x, dx, dy)) / A.diag(dx, dy)

    return smooth


def restrict(orig, out=None, avg=False):
    """
    Coarsen the original onto a coarser mesh

    Inputs
    ------

    orig[nx,ny] - A 2D numpy array. Each dimension must have
                  a size (2^n + 1) though nx != ny is possible

    Returns
    -------

    A 2D numpy array of size [(nx-1)/2+1, (ny-1)/2+1]
    """

    nx = orig.shape[0]
    ny = orig.shape[1]

    if (nx - 1) % 2 == 1 or (ny - 1) % 2 == 1:
        # Can't divide any further
        if out is None:
            return orig
        out.resize(orig.shape)
        out[:, :] = orig
        return

    # Dividing x and y in 2
    nx = (nx - 1) // 2 + 1
    ny = (ny - 1) // 2 + 1

    if out is None:
        out = zeros([nx, ny])
    else:
        out.resize([nx, ny])

    for x in range(1, nx - 1):
        for y in range(1, ny - 1):
            x0 = 2 * x
            y0 = 2 * y
            out[x, y] = orig[x0, y0] / 4.0
            +(
                orig[x0 + 1, y0]
                + orig[x0 - 1, y0]
                + orig[x0, y0 + 1]
                + orig[x0, y0 - 1]
            ) / 8.0
            +(
                orig[x0 - 1, y0 - 1]
                + orig[x0 - 1, y0 + 1]
                + orig[x0 + 1, y0 - 1]
                + orig[x0 + 1, y0 + 1]
            ) / 16.0
    if not avg:
        out *= 4.0

    return out


def interpolate(orig, out=None):
    """
    Interpolate a solution onto a finer mesh
    """
    nx = orig.shape[0]
    ny = orig.shape[1]

    nx2 = 2 * (nx - 1) + 1
    ny2 = 2 * (ny - 1) + 1

    if out is None:
        out = zeros([nx2, ny2])
    else:
        out[:, :] = 0.0

    for x in range(1, nx - 1):
        for y in range(1, ny - 1):
            x0 = 2 * x
            y0 = 2 * y

            out[x0 - 1, y0 - 1] += 0.25 * orig[x, y]
            out[x0 - 1, y0] += 0.5 * orig[x, y]
            out[x0 - 1, y0 + 1] += 0.25 * orig[x, y]

            out[x0, y0 - 1] += 0.5 * orig[x, y]
            out[x0, y0] = orig[x, y]
            out[x0, y0 + 1] += 0.5 * orig[x, y]

            out[x0 + 1, y0 - 1] += 0.25 * orig[x, y]
            out[x0 + 1, y0] += 0.5 * orig[x, y]
            out[x0 + 1, y0 + 1] += 0.25 * orig[x, y]

    return out


def smoothVcycle(A, x, b, dx, dy, niter=10, sublevels=0, direct=True):
    """
    Perform smoothing using multigrid


    """

    # Smooth
    for i in range(niter):
        x = smoothJacobi(A, x, b, dx, dy)

    if sublevels > 0:
        # Calculate the error
        error = b - A(x, dx, dy)

        # Restrict error onto coarser mesh
        Cerror = restrict(error)

        # smooth this error
        Cx = zeros(Cerror.shape)
        Cx = smoothVcycle(A, Cx, Cerror, dx * 2.0, dy * 2.0, niter, sublevels - 1)

        # Prolong the solution
        xupdate = interpolate(Cx)

        x = x + xupdate

    # Smooth
    for i in range(niter):
        x = smoothJacobi(A, x, b, dx, dy)

    return x


def smoothMG(A, x, b, dx, dy, niter=10, sublevels=1, ncycle=2):
    error = b - A(x, dx, dy)
    print("Starting max residual: %e" % (max(abs(error)),))

    for c in range(ncycle):
        x = smoothVcycle(A, x, b, dx, dy, niter, sublevels)

        error = b - A(x, dx, dy)
        print(
            "Cycle %d : %e"
            % (
                c,
                max(abs(error)),
            )
        )
    return x


class LaplacianOp:
    """
    Implements a simple Laplacian operator
    for use with the multigrid solver
    """

    def __call__(self, f, dx, dy):
        nx = f.shape[0]
        ny = f.shape[1]

        b = zeros([nx, ny])

        for x in range(1, nx - 1):
            for y in range(1, ny - 1):
                # Loop over points in the domain

                b[x, y] = (f[x - 1, y] - 2 * f[x, y] + f[x + 1, y]) / dx ** 2 + (
                    f[x, y - 1] - 2 * f[x, y] + f[x, y + 1]
                ) / dy ** 2

        return b

    def diag(self, dx, dy):
        return -2.0 / dx ** 2 - 2.0 / dy ** 2


class LaplaceSparse:
    def __init__(self, Lx, Ly):
        self.Lx = Lx
        self.Ly = Ly

    def __call__(self, nx, ny):
        dx = self.Lx / (nx - 1)
        dy = self.Ly / (ny - 1)

        # Create a linked list sparse matrix
        N = nx * ny
        A = eye(N, format="lil")
        for x in range(1, nx - 1):
            for y in range(1, ny - 1):
                row = x * ny + y
                A[row, row] = -2.0 / dx ** 2 - 2.0 / dy ** 2

                # y-1
                A[row, row - 1] = 1.0 / dy ** 2

                # y+1
                A[row, row + 1] = 1.0 / dy ** 2

                # x-1
                A[row, row - ny] = 1.0 / dx ** 2

                # x+1
                A[row, row + ny] = 1.0 / dx ** 2
        # Convert to Compressed Sparse Row (CSR) format
        return A.tocsr()


if __name__ == "__main__":

    # Test case

    from numpy import meshgrid, exp, linspace
    import matplotlib.pyplot as plt

    from timeit import default_timer as timer

    nx = 65
    ny = 65

    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)

    xx, yy = meshgrid(linspace(0, 1, nx), linspace(0, 1, ny))

    rhs = exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.4 ** 2)

    rhs[0, :] = 0.0
    rhs[:, 0] = 0.0
    rhs[nx - 1, :] = 0.0
    rhs[:, ny - 1] = 0.0

    x = zeros([nx, ny])

    x2 = x.copy()

    A = LaplacianOp()

    ################ SIMPLE ITERATIVE SOLVER ##############

    for i in range(1):
        x2 = smoothJacobi(A, x, rhs, dx, dy)
        x, x2 = x2, x  # Swap arrays

        error = rhs - A(x, dx, dy)
        print("%d : %e" % (i, max(abs(error))))

    ################ MULTIGRID SOLVER #######################

    print("Python multigrid solver")

    x = zeros([nx, ny])

    start = timer()
    x = smoothMG(A, x, rhs, dx, dy, niter=5, sublevels=3, ncycle=2)
    end = timer()

    error = rhs - A(x, dx, dy)
    print("Max error : {0}".format(max(abs(error))))
    print("Run time  : {0} seconds".format(end - start))

    ################ SPARSE MATRIX ##########################

    print("Sparse matrix solver")

    x2 = zeros([nx, ny])

    start = timer()
    solver = createVcycle(
        nx, ny, LaplaceSparse(1.0, 1.0), ncycle=2, niter=5, nlevels=4, direct=True
    )

    start_solve = timer()
    x2 = solver(x2, rhs)

    end = timer()

    error = rhs - A(x2, dx, dy)
    print("Max error : {0}".format(max(abs(error))))
    print(
        "Setup time: {0}, run time: {1} seconds".format(
            start_solve - start, end - start_solve
        )
    )

    print("Values: {0}, {1}".format(x2[10, 20], x[10, 20]))

    f = plt.figure()
    # plt.contourf(x)
    plt.plot(x[:, 32])
    plt.plot(x2[:, 32])
    plt.show()
