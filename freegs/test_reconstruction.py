from freegs import machine, equilibrium, reconstruction, plotting, critical
import numpy as np
import math
np.set_printoptions(threshold=np.inf)

"""
Test script running through 11 different equilibria
Ensures that each converges, and that psi at isoflux's are the same
"""

def test_reconstruction():
    alpha_n_list = [1,1,1,1,2,2,2,2,2,2,2]
    pprime_order_list = [2,2,2,2,3,3,3,3,3,3,3]
    ffprime_order_list = pprime_order_list
    x_point_list1 = [0.6,0.7,0.8,0.8,0.6,0.7,0.75,0.7,0.75,0.8,0.6]
    x_point_list2 = [-0.6,-0.6,-0.6,-0.8,-0.6,-0.6,-0.6,-0.7,-0.7,-0.7,-0.8]

    # Defining equilibrium grid
    Rmin = 0.1
    Rmax = 2
    Zmin = -1
    Zmax = 1
    nx = 65
    ny = 65
    alpha_m=1

    tokamak = machine.EfitTestMachine()
    Recon = reconstruction.Reconstruction(tokamak,0,0, show=False, tolerance=1e-15, VesselCurrents=False)

    for alpha_n, pprime_order, ffprime_order, x_z1, x_z2 in zip(alpha_n_list,pprime_order_list,ffprime_order_list,x_point_list1,x_point_list2):
        Recon.tokamak = machine.EfitTestMachine()
        Recon.pprime_order = pprime_order
        Recon.ffprime_order = ffprime_order

        reconstruction.generate_Measurements(tokamak=tokamak, alpha_m=alpha_m, alpha_n=alpha_n, x_z1=x_z1, x_z2=x_z2, show=False)

        measurement_dict = {}
        sigma_dict = {}
        for sensor in tokamak.sensors:
            measurement_dict[sensor.name]=sensor.measurement
            sigma_dict[sensor.name]=sensor.measurement/sensor.weight

        for name,coil in tokamak.coils:
            measurement_dict[name]=coil.current
            sigma_dict[name]=1e-5

        Recon.solve_from_dictionary(measurement_dict,sigma_dict)

        assert Recon.chi <=1e-3


def test_vessel_eigenmode():
    np.set_printoptions(threshold=np.inf)
    show = True
    import matplotlib.pyplot as plt

    # Defining equilibrium grid
    Rmin = 0.1
    Rmax = 2
    Zmin = -1
    Zmax = 1
    nx = 65
    ny = 65

    # Defining initial model and reconstruction parameters
    alpha_m = 1
    alpha_n = 2
    pprime_order = 3
    ffprime_order = 3
    x_z1 = 0.6
    x_z2 = -0.6
    x_r1 = 1.1
    x_r2 = 1.1

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

        if isinstance(fil, machine.Filament_Group):
            for subfil in fil.filaments:
                r, z = subfil.R, subfil.Z
                size = (eigenfunction[fil_num]) * 400
                tokamak.coils.append(('fil'+str(fil_num),machine.Coil(r, z, control=False, current=size)))
                fil_num += 1

    # Making up some measurements
    eq = reconstruction.generate_Measurements(tokamak=tokamak, alpha_m=alpha_m,
                                         alpha_n=alpha_n, x_z1=x_z1, x_z2=x_z2, x_r1=x_r1, x_r2=x_r2,
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
    Recon = reconstruction.Reconstruction(tokamak, pprime_order, ffprime_order, show=show)
    Recon.solve_from_dictionary(measurement_dict=measurement_dict, sigma_dict=sigma_dict)

    eig_coef = Recon.c[-tokamak.eigenbasis.shape[1]+i]
    print(Recon.c)
    assert abs(Recon.c[-tokamak.eigenbasis.shape[1]+i]) > 50*abs(Recon.c[-tokamak.eigenbasis.shape[1]+i+1]) and abs(Recon.c[-tokamak.eigenbasis.shape[1]+i]) > 50*abs(Recon.c[-tokamak.eigenbasis.shape[1]+i-1])


def advanced_reconstruction_test():
    import pickle
    Recon = reconstruction.Reconstruction(machine.EfitTestMachine(), 3,3, test=True, show=True)#, CurrentInitialisation=False)
    tokamaklist = ['DD.pkl', 'DS.pkl', 'LD.pkl', 'LS.pkl']
    for status in [False, True]:
        Recon.VesselCurrents = status
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
    Recon.plot()

def advanced_reconstruction_eigenmode():
    import pickle
    Recon = reconstruction.Reconstruction(machine.EfitTestMachine(), 3,3, test=True, show=True)
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
    Recon.plot()

advanced_reconstruction_test()
advanced_reconstruction_eigenmode()
#test_vessel_eigenmode()
