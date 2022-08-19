from freegs import machine, equilibrium, reconstruction, recon_tools, plotting, recon_matrices, critical
import numpy as np
import math
np.set_printoptions(threshold=np.inf)

"""
Test script running through 11 different equilibria
Ensures that each converges, and that psi at isoflux's are the same
"""

def test_reconstruction():
    alpha_n = [1,1,1,1,2,2,2,2,2,2,2]
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
    eq = equilibrium.Equilibrium(tokamak, Rmin=Rmin, Rmax=Rmax, Zmin=Zmin,Zmax=Zmax, nx=nx, ny=ny)
    G=recon_matrices.get_G(tokamak, eq)
    Gc = recon_matrices.get_Gc(tokamak,eq)
    show = True

    for alpha_n, pprime_order, ffprime_order, x_z1, x_z2 in zip(alpha_n,pprime_order_list,ffprime_order_list,x_point_list1,x_point_list2):

        # Making up some measurements
        sensors, coils, Beta0, L, mask, eq = recon_tools.get_values(tokamak, alpha_m, alpha_n, Rmin=Rmin, Rmax=Rmax, Zmin=Zmin, Zmax=Zmax,nx=nx,ny=ny,x_z1=x_z1,x_z2=x_z2, show=show)

        psi1 = eq.psi()
        opt, xpt = critical.find_critical(eq.R, eq.Z, psi1)

        M, sigma = recon_tools.give_values(tokamak, sensors, coils)

        # Creating Equilibrium
        eq = equilibrium.Equilibrium(tokamak, Rmin=Rmin, Rmax=Rmax, Zmin=Zmin, Zmax=Zmax,nx=nx,ny=ny,)

        # Performing Reconstruction
        chi,eq ,c = reconstruction.solve(tokamak, eq, M, sigma, pprime_order, ffprime_order, Fscale=False, tolerance=1e-9, show=show, VC=True, returnChi=True, G=G, Gc=Gc)

        psi2 = eq.psi()
        recon_opt, recon_xpt = critical.find_critical(eq.R, eq.Z, psi2)

        assert chi <=1
        assert math.isclose(opt[0][2], recon_opt[0][2], abs_tol=0.01)
        assert math.isclose(xpt[0][2],recon_xpt[0][2], abs_tol=0.01)


def test_vessel_eigenmode():
    from freegs import machine, equilibrium, reconstruction, recon_tools, recon_matrices, plotting
    import numpy as np
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

    # Generating Values

    # Creating Machine
    tokamak = machine.EfitTestMachine(createVessel=True, Nfils=50)

    eq = equilibrium.Equilibrium(tokamak)

    i = 4
    eigenfunction = tokamak.eigenbasis[:, i]

    fil_num = 0
    for fil in tokamak.vessel:
        if isinstance(fil,machine.filament):
            r, z = fil.R, fil.Z
            size = (eigenfunction[fil_num]) * 400
            tokamak.coils.append(('fil' + str(fil_num), machine.Coil(r, z, control=False, current=size)))
            fil_num+=1

        if isinstance(fil, machine.group_of_filaments):
            for subfil in fil.filaments:
                r, z = subfil.R, subfil.Z
                size = (eigenfunction[fil_num]) * 400
                tokamak.coils.append(('fil'+str(fil_num),machine.Coil(r, z, control=False, current=size)))
                fil_num += 1

    # Making up some measurements
    sensors, coils, Beta0, L, mask, eq1 = recon_tools.get_values(tokamak,
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

    # Creating Reconstruction Matrices
    G = recon_matrices.get_G(tokamak, eq)
    Gc = recon_matrices.get_Gc(tokamak, eq)
    Gfil = recon_matrices.get_Gfil(tokamak, eq)

    # Performing Reconstruction
    chi, eq, c = reconstruction.solve(tokamak, eq, M, sigma, pprime_order,
                                    ffprime_order, tolerance=1e-9, Fscale=True,
                                    VC=True, show=show, CI=True,
                                    returnChi=True,
                                    check_limited=check_limited,
                                    VesselCurrents=True,
                                    G=G, Gc=Gc, Gfil=Gfil, J=tokamak.eigenbasis)

    eig_coef = c[-tokamak.eigenbasis.shape[1]+i]
    assert abs(c[-tokamak.eigenbasis.shape[1]+i]) > 50*abs(c[-tokamak.eigenbasis.shape[1]+i+1]) and abs(c[-tokamak.eigenbasis.shape[1]+i]) > 50*abs(c[-tokamak.eigenbasis.shape[1]+i-1])

test_reconstruction()
