from freegs import machine, reconstruction
import numpy as np
import pickle

def test_reconstruction_advanced_novessel():
    """
    Test script running through 4 different equilibria
    Vessel currents turned off
    Tests for convergence and all correct sensors
    """
    eq_setup = {'tokamak': machine.EfitTestMachine(), 'Rmin': 0.1, 'Rmax': 2, 'Zmin': -1, 'Zmax': 1, 'nx': 65, 'ny': 65}

    Recon = reconstruction.Reconstruction(3,3, test=True, show=False, **eq_setup)
    tokamaklist = ['DD.pkl', 'DS.pkl', 'LD.pkl', 'LS.pkl']
    Recon.use_VesselCurrents = False

    for tokamakfile in tokamaklist:
        with open(tokamakfile, 'rb') as inp:
            tokamakfordict = pickle.load(inp)

        measurement_dict = {}
        sigma_dict = {}
        for sensor in tokamakfordict.sensors:
            measurement_dict[sensor.name] = sensor.measurement
            sigma_dict[sensor.name] = sensor.measurement / sensor.weight

        for coil_name, coil in tokamakfordict.coils:
            measurement_dict[coil_name] = coil.current
            sigma_dict[coil_name] = 1e-5

        Recon.solve_from_dictionary(measurement_dict,sigma_dict)

        print('Passed')

def test_reconstruction_advanced_vessel():
    """
    Test script running through 4 different equilibria
    Vessel currents on
    Tests for convergence and all correct sensors
    """
    eq_setup = {'tokamak': machine.EfitTestMachine(), 'Rmin': 0.1, 'Rmax': 2, 'Zmin': -1, 'Zmax': 1, 'nx': 65, 'ny': 65}
    Recon = reconstruction.Reconstruction(3,3, test=True, show=False, **eq_setup)
    tokamaklist = ['DD.pkl', 'DS.pkl', 'LD.pkl', 'LS.pkl']
    Recon.use_VesselCurrents = True
    for tokamakfile in tokamaklist:
        with open(tokamakfile, 'rb') as inp:
            tokamakfordict = pickle.load(inp)

        measurement_dict = {}
        sigma_dict = {}
        for sensor in tokamakfordict.sensors:
            measurement_dict[sensor.name] = sensor.measurement
            sigma_dict[sensor.name] = sensor.measurement / sensor.weight

        for coil_name, coil in tokamakfordict.coils:
            measurement_dict[coil_name] = coil.current
            sigma_dict[coil_name] = 1e-5

        Recon.solve_from_dictionary(measurement_dict,sigma_dict)

def test_advanced_eigenmode():
    """
    Test script running through 4 different equilibria
    Vessel currents excited
    Tests for convergence and all correct sensors
    """
    eq_setup = {'tokamak': machine.EfitTestMachine(), 'Rmin': 0.1, 'Rmax': 2, 'Zmin': -1, 'Zmax': 1, 'nx': 65, 'ny': 65}
    Recon = reconstruction.Reconstruction(3,3, test=True, show=False, **eq_setup)
    tokamaklist2 = ['DDV.pkl', 'DSV.pkl', 'LDV.pkl', 'LSV.pkl']

    for tokamakfile in tokamaklist2:
        with open(tokamakfile, 'rb') as inp:
            tokamakfordict2 = pickle.load(inp)

        measurement_dict = {}
        sigma_dict = {}
        for sensor in tokamakfordict2.sensors:
            measurement_dict[sensor.name] = sensor.measurement
            sigma_dict[sensor.name] = sensor.measurement / sensor.weight

        for coil_name, coil in tokamakfordict2.coils:
            measurement_dict[coil_name] = coil.current
            sigma_dict[coil_name] = 1e-5

        Recon.solve_from_dictionary(measurement_dict, sigma_dict)
        print('Passed')

def test_vessel_eigenmode():
    """
    Test script that creates an equilibrium with a specific vessel mode excited, then tests
    the reonctsurctions ability to isolate the mode
    """
    np.set_printoptions(threshold=np.inf)
    show = False
    import matplotlib.pyplot as plt
    eq_setup = {'tokamak': machine.EfitTestMachine(), 'Rmin': 0.1, 'Rmax': 2, 'Zmin': -1, 'Zmax': 1, 'nx': 65, 'ny': 65}

    # Defining initial model and reconstruction parameters
    pprime_order = 3
    ffprime_order = 3
    x_z1 = 0.6
    x_z2 = -0.6
    x_r1 = 1
    x_r2 = 1

    # Creating Machine
    tokamak = machine.EfitTestMachine()
    i = 4
    eigenfunction = tokamak.eigenbasis[:, i]

    fil_num = 0
    for fil in tokamak.vessel:

        if isinstance(fil,machine.Filament):
            r, z = fil.R, fil.Z
            size = (eigenfunction[fil_num]) * 400
            tokamak.coils.append(('fil' + str(fil_num), machine.Coil(r, z, control=False, current=size)))
            fil_num+=1

        if isinstance(fil, machine.Passive):
            for subfil in fil.filaments:
                r, z = subfil.R, subfil.Z
                size = (eigenfunction[fil_num]) * 400
                tokamak.coils.append(('fil'+str(fil_num),machine.Coil(r, z, control=False, current=size)))
                fil_num += 1

    # Making up some measurements
    eq = reconstruction.generate_Measurements(tokamak=tokamak, alpha_m=1,
                                         alpha_n=2, x_z1=x_z1, x_z2=x_z2, x_r1=x_r1, x_r2=x_r2,
                                         show=show)

    measurement_dict = {}
    sigma_dict = {}
    for sensor in tokamak.sensors:
        measurement_dict[sensor.name] = sensor.measurement
        sigma_dict[sensor.name] = sensor.measurement / sensor.weight

    for name, coil in tokamak.coils:
        measurement_dict[name] = coil.current
        sigma_dict[name] = 1e-5

    # Reconstruction
    tokamak = machine.EfitTestMachine()
    Recon = reconstruction.Reconstruction(pprime_order, ffprime_order, show=show, **eq_setup)
    Recon.solve_from_dictionary(measurement_dict=measurement_dict, sigma_dict=sigma_dict)
    assert abs(Recon.coefs[-tokamak.eigenbasis.shape[1]+i]) > 50*abs(Recon.coefs[-tokamak.eigenbasis.shape[1]+i+1]) and abs(Recon.coefs[-tokamak.eigenbasis.shape[1]+i]) > 50*abs(Recon.coefs[-tokamak.eigenbasis.shape[1]+i-1])

