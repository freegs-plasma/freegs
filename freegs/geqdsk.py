"""
Handles reading and writing of Equilibrium objects
"""

from . import _geqdsk
from . import critical
from .equilibrium import Equilibrium
from . import jtor
from . import control

from scipy import interpolate
from numpy import linspace, amin, amax, reshape, ravel, zeros

def write(eq, fh, label=None, oxpoints=None, fileformat=_geqdsk.write):
    """
    Write a GEQDSK equilibrium file, given a FreeGS Equilibrium object
    
    eq - Equilibrium object
    fh - file handle
    
    label - Text label to put in the file
    oxpoints - O- and X-points  (opoint, xpoint) returned by critical.find_critical
    """
    # Get poloidal flux
    psi = eq.psi()

    # Get size of the grid
    nx,ny = psi.shape

    if oxpoints:
        opoint, xpoint = oxpoints
    else:
        # Find the O- and X-points
        opoint, xpoint = critical.find_critical(eq.R, eq.Z, psi)

    rmin = eq.Rmin
    rmax = eq.Rmax
    zmin = eq.Zmin
    zmax = eq.Zmax

    fvac = eq.fpolVac() # Vacuum f = R*Bt
    R0 = 1.0 # Reference location
    B0 = fvac / R0 # Reference vacuum toroidal magnetic field
    
    data = {"nx":nx, "ny":ny,
            "rdim":rmax-rmin, # Horizontal dimension in meter of computational box
            "zdim":zmax-zmin, # Vertical dimension in meter of computational box
            "rcentr":R0, # R in meter of vacuum toroidal magnetic field BCENTR
            "bcentr":fvac/R0, # Vacuum magnetic field at rcentr
            "rleft":rmin, # Minimum R in meter of rectangular computational box
            "zmid":0.5*(zmin + zmax)} # Z of center of computational box in meter
            
    
    data["rmagx"], data["zmagx"], data["simagx"] = opoint[0] # magnetic axis
    
    data["sibdry"] = xpoint[0][2]  # Psi at boundary
    
    data["cpasma"] = eq.plasmaCurrent() # Plasma current [A]

    psinorm = linspace(0.0, 1.0, nx, endpoint=False) # Does not include separatrix

    data["fpol"] = eq.fpolPsiN(psinorm)
    data["pres"] = eq.pressurePsiN(psinorm)

    data["psi"] = psi
    
    qpsi = zeros([nx])
    qpsi[1:] = eq.qPsiN(psinorm[1:]) # Exclude axis
    qpsi[0] = qpsi[1]
    data["qpsi"] = qpsi
    
    if hasattr(eq, "rlim") and hasattr(eq, "zlim"):
        data["rlim"] = eq.rlim
        data["zlim"] = eq.zlim
    
    # Call fileformat to write the data
    fileformat(data, fh, label=label)

import matplotlib.pyplot as plt

def read(fh, machine):
    """
    Reads a G-EQDSK format file
    
    fh - file handle
    machine - a Machine object defining coil locations
    """

    # Read the data as a Dictionary
    data = _geqdsk.read(fh)

    # Create an Equilibrium object
    eq = Equilibrium(tokamak = machine,
                     Rmin = data["rleft"],
                     Rmax = data["rleft"] + data["rdim"],
                     Zmin = data["zmid"] - 0.5*data["zdim"],
                     Zmax = data["zmid"] + 0.5*data["zdim"],
                     nx=data["nx"], ny=data["ny"],         # Number of grid points
                     fvac=data["rcentr"] * data["bcentr"] # Vacuum f=R*Bt
    )

    # Range of psi normalises psi derivatives
    psirange = data["sibdry"] - data["simagx"]
    
    psinorm = linspace(0.0, 1.0, data["nx"], endpoint=True)
    
    # Create a spline fit to pressure, f and f**2
    p_spl = interpolate.InterpolatedUnivariateSpline(psinorm, data["pres"])
    pprime_spl = interpolate.InterpolatedUnivariateSpline(psinorm, data["pres"] / psirange).derivative()
    
    f_spl = interpolate.InterpolatedUnivariateSpline(psinorm, data["fpol"])
    ffprime_spl = interpolate.InterpolatedUnivariateSpline(psinorm, 0.5*data["fpol"]**2/psirange).derivative() 

    # functions to return p, pprime, f and ffprime
    def p_func(psinorm):
        return reshape(p_spl(ravel(psinorm)),psinorm.shape)

    def f_func(psinorm):
        return reshape(f_spl(ravel(psinorm)),psinorm.shape)

    def pprime_func(psinorm):
        return reshape(pprime_spl(ravel(psinorm)),psinorm.shape)
    
    def ffprime_func(psinorm):
        return reshape(ffprime_spl(ravel(psinorm)),psinorm.shape)

    
    # Create a set of profiles to calculate toroidal current density Jtor
    profiles = jtor.ProfilesPprimeFfprime(pprime_func,
                                          ffprime_func,
                                          p_func=p_func, f_func=f_func)


    # Calculate Jtor using input psi
    Jtor = profiles(eq.R, eq.Z, data["psi"])
    
    # Use this Jtor to calculate plasma psi
    eq.solve(Jtor, niter=10, sublevels=5, ncycle=5)

    print("Plasma current: {0} Amps, input: {1} Amps".format(eq.plasmaCurrent(), data["cpasma"]))
    
    # Identify points to constrain: X-points, O-points
    opoint, xpoint = critical.find_critical(eq.R, eq.Z, data["psi"])
    
    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.set_aspect('equal')
    axis.set_xlabel("Major radius [m]")
    axis.set_ylabel("Height [m]")
    
    levels = linspace(amin(data["psi"]), amax(data["psi"]), 50)
    axis.contour(eq.R,eq.Z,data["psi"], levels=levels, colors='k')
    for r,z,_ in xpoint:
        axis.plot(r,z,'ro')
    for r,z,_ in opoint:
        axis.plot(r,z,'go')
    
    psivals = xpoint # + [opoint[0]]
    #psivals = [opoint[0], xpoint[0], xpoint[1]]

    # Find best fit for coil currents
    control.constrain(eq, xpoints=xpoint, psivals=psivals,gamma=1e-14)
    
    psi = eq.psi()
    #levels = linspace(amin(psi), amax(psi), 100)
    axis.contour(eq.R,eq.Z, psi, levels=levels, colors='r')
    plt.show()
    
    return eq
