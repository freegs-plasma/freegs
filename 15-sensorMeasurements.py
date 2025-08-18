from freegs import boundary, control, equilibrium, jtor, machine, solve

tokamak = machine.TestTokamakSensor()

eq = equilibrium.Equilibrium(tokamak=tokamak,
                        Rmin=0.1, Rmax=2.0,
                        Zmin=-1.0, Zmax=1.0,
                        nx=65, ny=65,
                        boundary=boundary.freeBoundaryHagenow)

profiles = jtor.ConstrainPaxisIp(eq, 1e3, 2e5, 2.0)

xpoints = [(1.1, -0.6), (1.1, 0.8)]
isoflux = [(1.1,-0.6, 1.1,0.6)]

constrain = control.constrain(xpoints=xpoints, isoflux=isoflux)

solve(eq, profiles, constrain, show=True)

tokamak.printCurrents()
tokamak.printMeasurements(eq=eq)
