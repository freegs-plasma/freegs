import freegs
import numpy as np
from scipy.special import ellipk, ellipe
from scipy.constants import mu_0

def test_offaxis_Bfield():
    """
    Using the analytical equation for off axis b field due to a single coil loop
    And comparing to result returned from running the sensors
    Note - although the equation is analytic, its evaluation requires the use
    of elliptic integrals, therefore cannot be said to be exact
    """
    r = 4
    z = 7
    theta = np.pi / 2
    Rcoil = 1.5
    zcoil = 1
    current = 10000
    coil = [('Coil', freegs.machine.Coil(Rcoil, zcoil, current=current))]
    sensor = [freegs.machine.PoloidalFieldSensor(r, z, theta)]
    tokamak = freegs.machine.Machine(coil, sensors=sensor)

    # first want to write a calc for the analytical B field (with elliptical integrals)
    # defining the constants
    B0 = (current * mu_0) / (
            2 * Rcoil)  # this is the field on axis centre of coil plane
    a = r / Rcoil
    b = (z - zcoil) / Rcoil
    c = (z - zcoil) / r
    Q = (1 + a) ** 2 + b ** 2
    k = ((4 * a) / Q) ** 0.5

    # calculating B components
    Bz = B0 * (1 / (np.pi * (Q ** 0.5))) * (ellipe(k ** 2) * (
            (1 - a ** 2 - b ** 2) / (Q - 4 * a)) + ellipk(k ** 2))
    Br = B0 * (c / (np.pi * (Q ** 0.5))) * (ellipe(k ** 2) * (
            (1 + a ** 2 + b ** 2) / (Q - 4 * a)) - ellipk(k ** 2))
    B = Bz * np.sin(theta) + Br * np.cos(theta)

    # secondly want to pull the magentic field i am calculating from the sensors
    for sensor in tokamak.sensors:
        if isinstance(sensor, freegs.machine.PoloidalFieldSensor):
            sensor.get_measure(tokamak, None)
            assert np.isclose(sensor.measurement, B, atol=1e-8)


def test_xpoint_field():
    """
    Runs a basic equilibrium with constraints, checks if the sensors measure
    the correct constraints value
    """
    tokamak = freegs.machine.TestTokamak()
    tokamak.sensors = [freegs.machine.PoloidalFieldSensor(1.1, -0.6, 0),
                       freegs.machine.PoloidalFieldSensor(1.1, 0.6, 0),
                       freegs.machine.PoloidalFieldSensor(1.1, -0.6, np.pi/2),
                       freegs.machine.PoloidalFieldSensor(1.1, 0.6, np.pi/2)
                       ]

    eq = freegs.Equilibrium(tokamak=tokamak,
                            Rmin=0.1, Rmax=2.0,  # Radial domain
                            Zmin=-1.0, Zmax=1.0,  # Height range
                            nx=65, ny=65,  # Number of grid points
                            boundary=freegs.boundary.freeBoundary)  # Boundary condition

    profiles = freegs.jtor.ConstrainPaxisIp(eq, 1e3,
                                            # Plasma pressure on axis [Pascals]
                                            2e5,  # Plasma current [Amps]
                                            2.0)  # Vacuum f=R*Bt

    xpoints = [(1.1, -0.6),  # (R,Z) locations of X-points
               (1.1, 0.6)]

    isoflux = [(1.1, -0.6, 1.1, 0.6)]  # (R1,Z1, R2,Z2) pair of locations

    constrain = freegs.control.constrain(xpoints=xpoints, isoflux=isoflux)

    constrain(eq)

    tokamak.takeMeasurements(eq)

    for sensor in tokamak.sensors:
        assert np.isclose(sensor.measurement, 0.0, atol=1e-5)


def test_iso_flux():
    """
    Similarly to earlier test, checks the 2 points of isoflux and ensures
    sensor meaurement confirms constraint
    """
    tokamak = freegs.machine.TestTokamak()
    tokamak.sensors = [freegs.machine.FluxLoopSensor(1.1, -0.6),
                       freegs.machine.FluxLoopSensor(1.1, 0.6)]

    eq = freegs.Equilibrium(tokamak=tokamak,
                            Rmin=0.1, Rmax=2.0,  # Radial domain
                            Zmin=-1.0, Zmax=1.0,  # Height range
                            nx=65, ny=65,  # Number of grid points
                            boundary=freegs.boundary.freeBoundary)  # Boundary condition

    profiles = freegs.jtor.ConstrainPaxisIp(eq, 1e3,
                                            # Plasma pressure on axis [Pascals]
                                            2e5,  # Plasma current [Amps]
                                            2.0)  # Vacuum f=R*Bt

    xpoints = [(1.1, -0.6),  # (R,Z) locations of X-points
               (1.1, 0.6)]

    isoflux = [(1.1, -0.6, 1.1, 0.6)]  # (R1,Z1, R2,Z2) pair of locations

    constrain = freegs.control.constrain(xpoints=xpoints, isoflux=isoflux)

    constrain(eq)

    tokamak.takeMeasurements(eq)

    assert np.isclose(tokamak.sensors[0].measurement, tokamak.sensors[1].measurement,
                        atol=1e-8)


def test_flux():
    """
    Testing analytical flux agaisnt the numerical sensor output for a single coil
    """
    r = 4
    z = 7
    theta = np.pi / 2
    Rcoil = 1.5
    zcoil = 1
    current = 100000
    coil = [('Coil', freegs.machine.Coil(Rcoil, zcoil, current=current))]
    sensor = [freegs.machine.FluxLoopSensor(r, z)]
    tokamak = freegs.machine.Machine(coil, sensors=sensor)

    # first want to write a calc for the analytical flux (with greens function)

    psi = freegs.gradshafranov.Greens(Rcoil, zcoil, r, z) * current

    for sensor in tokamak.sensors:
        if isinstance(sensor, freegs.machine.FluxLoopSensor):
            sensor.get_measure(tokamak, None)
            assert np.isclose(sensor.measurement, psi, atol=1e-8)


def test_rog_around_coil():
    """
    Test single Rog around a coil
    """

    Rcoil = 1.5
    zcoil = 1
    zcoil2 = -1
    current = 100000.0
    coil = [('Coil 1', freegs.machine.Coil(Rcoil, zcoil, current=current)),
            ('Coil 2', freegs.machine.Coil(Rcoil, zcoil2, current=current))]
    sensor = [freegs.machine.RogowskiSensor([1.45, 1.45, 1.55, 1.55],
                                            [0.95, 1.05, 1.05, 0.95])]
    tokamak = freegs.machine.Machine(coil, sensors=sensor)

    tokamak.takeMeasurements()

    assert np.isclose(tokamak.sensors[0].measurement, current, atol=1)


def test_rog_around_Shapedcoil():
    """
    Test single Rog around a coil
    """

    Rcoil = 1.5
    zcoil = 1
    current = 100000.0
    coil = [('Coil 1', freegs.machine.ShapedCoil(
        [(0.95, -0.1), (0.95, 0.1), (1.05, 0.1), (1.05, -0.1)],
        current=current))]
    sensor = [freegs.machine.RogowskiSensor([0.9, 0.9, 1.0, 1.0],
                                            [-0.2, 0.2, 0.2, -0.2])]
    tokamak = freegs.machine.Machine(coil, sensors=sensor)

    tokamak.takeMeasurements()
    for sensor in tokamak.sensors:
        assert np.isclose(sensor.measurement, current / 2, atol=1)


def test_rog_around_Filamentcoil():

    tokamak = freegs.machine.EmptyTokamak()
    total_current = 300

    tokamak.coils = [('FILCOIL1',freegs.machine.FilamentCoil((0.5,0.5,0.5),(0.9,1,1.1), current=total_current)),('FILCOIL2',freegs.machine.FilamentCoil((0.5,0.5,0.5),(-0.9,-1,-1.1), current=total_current))]
    tokamak.sensors = [freegs.machine.RogowskiSensor([0.4,0.4,0.6,0.6], [0.8,1.2,1.2,0.8], name='UPPERCOIL'),freegs.machine.RogowskiSensor([0.4,0.4,0.6,0.6], [0.95,1.05,1.05,0.95], name='UPPERFIL')]

    tokamak.takeMeasurements()

    assert tokamak.sensors[0].measurement == total_current and tokamak.sensors[1].measurement == total_current / 3


def test_rog_around_circuit():

    tokamak = freegs.machine.EmptyTokamak()
    circuit_current = 100
    npoints = 3

    tokamak.coils = [('CIRCUIT', freegs.machine.Circuit([('FILCOIL1',freegs.machine.FilamentCoil((0.5,0.5,0.5),(0.9,1,1.1)), npoints),('FILCOIL2',freegs.machine.FilamentCoil((0.5,0.5,0.5),(-0.9,-1,-1.1)), npoints)], current=circuit_current))]
    tokamak.sensors = [freegs.machine.RogowskiSensor([0.4,0.4,0.6,0.6], [0.8,1.2,1.2,0.8], name='UPPERCOIL'),freegs.machine.RogowskiSensor([0.4,0.4,0.6,0.6], [0.95,1.05,1.05,0.95], name='UPPERFIL')]

    tokamak.takeMeasurements()

    assert tokamak.sensors[0].measurement == circuit_current * npoints and tokamak.sensors[1].measurement == circuit_current


def test_rog_with_plasma():
    """
    Testing Rog around a plasma profile
    """
    tokamak = freegs.machine.EmptyTokamak()  #
    tokamak.sensors = [
        freegs.machine.RogowskiSensor([0.1, 0.1, 2, 2], [-1, 1, 1, -1])]

    eq = freegs.Equilibrium(tokamak=tokamak,
                            Rmin=0.1, Rmax=2.0,  # Radial domain
                            Zmin=-1.0, Zmax=1.0,  # Height range
                            nx=65, ny=65,  # Number of grid points
                            boundary=freegs.boundary.freeBoundary)  # Boundary condition

    plasmacurrent = 300000
    profiles = freegs.jtor.ConstrainPaxisIp(eq, 1e3,
                                            # Plasma pressure on axis [Pascals]
                                            plasmacurrent,
                                            # Plasma current [Amps]
                                            2.0)  # Vacuum f=R*Bt

    eq.solve(profiles)
    tokamak.takeMeasurements(eq)
    print(tokamak.sensors[0].measurement)
    assert np.isclose(tokamak.sensors[0].measurement, plasmacurrent,atol=1000)
