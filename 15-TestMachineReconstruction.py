from freegs import machine, equilibrium, reconstruction, recon_tools, recon_matrices, plotting
import numpy as np
show = True
check_limited = True
np.set_printoptions(threshold=np.inf)

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
x_z1 = -0.6
x_z2 = 0.7
x_r1 = 1.1
x_r2 = 1.1

tokamak = machine.EfitTestMachine(createVessel=True, Nfils=100)

# Making up some measurements
sensors, coils, eq1 = recon_tools.get_values(tokamak,alpha_m,alpha_n,Rmin=Rmin,Rmax=Rmax,Zmin=Zmin,Zmax=Zmax,nx=nx, ny=ny,x_z1=x_z1,x_z2=x_z2,x_r1=x_r1,x_r2=x_r2,show=show,check_limited=check_limited)

M, sigma = recon_tools.give_values(tokamak, tokamak.sensors, tokamak.coils)

# Creating Equilibrium
eq = equilibrium.Equilibrium(tokamak, Rmin=Rmin, Rmax=Rmax, Zmin=Zmin, Zmax=Zmax,nx=nx,ny=ny)

# Performing Reconstruction
print('Starting Reconstruction')
chi, eq2, c = reconstruction.solve(tokamak, eq, M, sigma, pprime_order, ffprime_order, tolerance=1e-7, Fscale=True,
                               VC=True, show=show, CI=True, returnChi=True, check_limited=check_limited, VesselCurrents=True,
                               G=tokamak.G, Gc=tokamak.Gc, Gfil=tokamak.Gfil, J=tokamak.eigenbasis)


plotting.plotEquilibrium(eq2)