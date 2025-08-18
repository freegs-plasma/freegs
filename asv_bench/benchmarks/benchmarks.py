from io import BytesIO

import freegs
from freegs import Equilibrium, fieldtracer, solve
from freegs.boundary import fixedBoundary, freeBoundaryHagenow
from freegs.control import constrain
from freegs.machine import TestTokamak

STANDARD_PARAMS = dict(Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0, nx=17, ny=17)


def ConstrainPaxisIp(*args):
    """Backwards compatibility shim, as the signature changed in 0.7.0"""
    eq, paxis, Ip, fvac = args

    if freegs.__version__ >= "0.7.0":
        return freegs.jtor.ConstrainPaxisIp(eq, paxis, Ip, fvac)
    return freegs.jtor.ConstrainPaxisIp(paxis, Ip, fvac)


def time_freeboundary_solve():
    eq = Equilibrium(
        tokamak=TestTokamak(), boundary=freeBoundaryHagenow, **STANDARD_PARAMS
    )

    profiles = ConstrainPaxisIp(eq, 1e3, 2e5, 2.0)
    xpoints = [(1.1, -0.6), (1.1, 0.8)]
    isoflux = [(1.1, -0.6, 1.1, 0.6)]
    control = constrain(xpoints=xpoints, isoflux=isoflux)
    solve(eq, profiles, control)


def time_fixedboundary_solve():
    eq = Equilibrium(boundary=fixedBoundary, **STANDARD_PARAMS)
    profiles = ConstrainPaxisIp(eq, 1e3, 1e5, 1.0)
    solve(eq, profiles)


def time_find_xpoints():
    tokamak = TestTokamak()
    eq = Equilibrium(tokamak=tokamak, **STANDARD_PARAMS)
    control = constrain(xpoints=[(1.1, -0.8), (1.1, 0.8)])
    control(eq)
    _ = eq.psi()


def time_sensors():
    tokamak = freegs.machine.TestTokamakSensor()
    eq = freegs.equilibrium.Equilibrium(
        tokamak=tokamak, boundary=freeBoundaryHagenow, **STANDARD_PARAMS
    )

    profiles = ConstrainPaxisIp(eq, 1e3, 2e5, 2.0)
    control = constrain(
        xpoints=[(1.1, -0.6), (1.1, 0.8)], isoflux=[(1.1, -0.6, 1.1, 0.6)]
    )
    solve(eq, profiles, control)


class TimeEquilibrium:
    """Various benchmarks that can reuse the same equilbrium instance"""

    def setup_cache(self):
        tokamak = TestTokamak()
        eq = Equilibrium(
            tokamak=tokamak, boundary=freeBoundaryHagenow, **STANDARD_PARAMS
        )
        profiles = ConstrainPaxisIp(eq, 1e4, 1e6, 2.0)
        xpoints = [(1.1, -0.6), (1.1, 0.8)]
        isoflux = [(1.1, -0.6, 1.1, 0.6)]
        control = constrain(xpoints=xpoints, isoflux=isoflux)
        solve(eq, profiles, control, maxits=25, atol=1e-3, rtol=1e-1)

        # SuperLU solver can't be pickled, but we don't need it for
        # these benchmarks
        eq.setSolver(None)
        return eq

    def time_dump(self, eq):
        memory_file = BytesIO()

        with freegs.OutputFile(memory_file, "w") as f:
            f.write_equilibrium(eq)

        with freegs.OutputFile(memory_file, "r") as f:
            _ = f.read_equilibrium()

    def time_find_critical(self, eq):
        _ = freegs.critical.find_critical(eq.R, eq.Z, eq.psi())

    def time_refine(self, eq):
        freegs.equilibrium.refine(eq)

    def time_trace_fieldlines(self, eq):
        fieldtracer.traceFieldLines(eq, nturns=1, nlines=1, npoints=10)
