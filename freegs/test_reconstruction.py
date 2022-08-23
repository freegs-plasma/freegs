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
    alpha_n = 2

    tokamak = machine.EfitTestMachine(createVessel=True)
    eq = equilibrium.Equilibrium(tokamak, Rmin=Rmin, Rmax=Rmax, Zmin=Zmin,Zmax=Zmax, nx=nx, ny=ny)
    Recon = reconstruction.Reconstruction(tokamak,0,0, eq=eq)
    Recon.generate_Greens()
    show = True

    for alpha_n, pprime_order, ffprime_order, x_z1, x_z2 in zip(alpha_n_list,pprime_order_list,ffprime_order_list,x_point_list1,x_point_list2):

        Recon.pprime_order = pprime_order
        Recon.ffprime_order = ffprime_order

        Recon.generate_Measurements(alpha_m=alpha_m, alpha_n=alpha_n, x_z1=x_z1, x_z2=x_z2)
        psi1 = Recon.eq.psi()

        opt, xpt = critical.find_critical(Recon.eq.R, Recon.eq.Z, psi1)

        Recon.take_Measurements_from_tokamak()
        Recon.solve()
        psi2 = Recon.eq.psi()
        recon_opt, recon_xpt = critical.find_critical(Recon.eq.R, Recon.eq.Z, psi2)

        assert Recon.chi <=1
        assert math.isclose(opt[0][2], recon_opt[0][2], abs_tol=0.01)
        assert math.isclose(xpt[0][2],recon_xpt[0][2], abs_tol=0.01)


def test_vessel_eigenmode():
    np.set_printoptions(threshold=np.inf)
    show = True
    check_limited = False
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
    x_z1 = 0.75
    x_z2 = -0.6
    x_r1 = 1.1
    x_r2 = 1.1


    # Creating Machine
    tokamak = machine.EfitTestMachine(createVessel=True)


    eq = equilibrium.Equilibrium(tokamak)

    for i in range(tokamak.eigenbasis.shape[1]):
        for j in range(tokamak.eigenbasis.shape[1]):
            if i == j :
                assert np.dot(tokamak.eigenbasis[:, i],tokamak.eigenbasis[:, j]) == 1
            else:
                assert np.dot(tokamak.eigenbasis[:, i],tokamak.eigenbasis[:, j]) == 0 # use math is close

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
    sensors, coils, eq1 = recon_tools.get_values(tokamak,
                                                                 alpha_m,
                                                                 alpha_n,
                                                                 Rmin=Rmin,
                                                                 Rmax=Rmax,
                                                                 Zmin=Zmin,
                                                                 Zmax=Zmax,
                                                                 nx=nx, ny=ny,
                                                                 x_z1=x_z1,
                                                                 x_z2=x_z2,
                                                                 x_r1=x_r1,
                                                                 x_r2=x_r2,
                                                                 show=show,
                                                                 check_limited=check_limited)

    # Reconstruction
    tokamak = machine.EfitTestMachine(createVessel=True)

    M, sigma = recon_tools.give_values(tokamak, sensors, coils)

    # Creating Equilibrium
    eq = equilibrium.Equilibrium(tokamak, Rmin=Rmin, Rmax=Rmax, Zmin=Zmin,
                                 Zmax=Zmax, nx=nx, ny=ny)

    # Performing Reconstruction
    chi, eq, c = reconstruction.solve(tokamak, eq, M, sigma, pprime_order,
                                    ffprime_order, tolerance=1e-9, FittingWeight=True,
                                    VerticalControl=True, show=show, CurrentInitialisation=True,
                                    returnChi=True,
                                    check_limited=check_limited,
                                    VesselCurrents=True)

    eig_coef = c[-tokamak.eigenbasis.shape[1]+i]
    print(c)
    assert abs(c[-tokamak.eigenbasis.shape[1]+i]) > 50*abs(c[-tokamak.eigenbasis.shape[1]+i+1]) and abs(c[-tokamak.eigenbasis.shape[1]+i]) > 50*abs(c[-tokamak.eigenbasis.shape[1]+i-1])

test_reconstruction()
test_vessel_eigenmode()
