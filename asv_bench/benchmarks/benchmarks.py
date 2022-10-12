import freegs


def time_freeboundary():
    eq = freegs.Equilibrium(
        tokamak=freegs.machine.TestTokamak(),
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-1.0,
        Zmax=1.0,
        nx=65,
        ny=65,
        boundary=freegs.boundary.freeBoundaryHagenow,
    )

    # Signature changed in 0.7.0
    if freegs.__version__ >= "0.7.0":
        profiles = freegs.jtor.ConstrainPaxisIp(eq, 1e3, 2e5, 2.0)
    else:
        profiles = freegs.jtor.ConstrainPaxisIp(1e3, 2e5, 2.0)

    xpoints = [(1.1, -0.6), (1.1, 0.8)]
    isoflux = [(1.1, -0.6, 1.1, 0.6)]
    constrain = freegs.control.constrain(xpoints=xpoints, isoflux=isoflux)
    freegs.solve(eq, profiles, constrain)
