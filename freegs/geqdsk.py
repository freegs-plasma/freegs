"""
Handles reading and writing of Equilibrium objects

Writing is relatively straightforward, but reading requires inferring
the currents in the PF coils

Copyright 2016 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk

This file is part of FreeGS.

FreeGS is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FreeGS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGS.  If not, see <http://www.gnu.org/licenses/>.

"""

from . import _geqdsk
from . import critical
from .equilibrium import Equilibrium
from . import jtor
from . import control
from . import picard

from scipy import interpolate
from numpy import linspace, amin, amax, reshape, ravel, zeros, argmax, clip, sin, cos, pi

def write(eq, fh, label=None, oxpoints=None, fileformat=_geqdsk.write):
    """
    Write a GEQDSK equilibrium file, given a FreeGS Equilibrium object
    
    eq - Equilibrium object
    fh - file handle
    
    label - Text label to put in the file
    oxpoints - O- and X-points  (opoint, xpoint) returned by critical.find_critical
          If not given, it will be calculated using critical.find_critical
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

    fvac = eq.fvac() # Vacuum f = R*Bt
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

    data["fpol"] = eq.fpol(psinorm)
    data["pres"] = eq.pressure(psinorm)

    data["psi"] = psi
    
    qpsi = zeros([nx])
    qpsi[1:] = eq.q(psinorm[1:]) # Exclude axis
    qpsi[0] = qpsi[1]
    data["qpsi"] = qpsi
    
    if hasattr(eq, "rlim") and hasattr(eq, "zlim"):
        data["rlim"] = eq.rlim
        data["zlim"] = eq.zlim
    
    # Call fileformat to write the data
    fileformat(data, fh, label=label)

import matplotlib.pyplot as plt

def read(fh, machine, rtol=1e-3, ntheta=8, show=False, axis=None):
    """
    Reads a G-EQDSK format file
    
    fh      - file handle
    machine - a Machine object defining coil locations
    rtol    - Relative error in nonlinear solve
    ntheta  - Number of points in poloidal angle on the separatrix
              this is used to constrain the plasma shape
    show    - Set to true to show solution in a new figure
    axis    - Set to an axis for plotting. Implies show=True
    
    A nonlinear solve will be performed, using Picard iteration
    
    Returns
    -------

    An Equilibrium object eq. In addition, the following is available:

    eq.control   - The control system
    eq._profiles - The profiles object
    
    """

    if axis is not None:
        show = True

    # Read the data as a Dictionary
    data = _geqdsk.read(fh)

    # Create an Equilibrium object
    eq = Equilibrium(tokamak = machine,
                     Rmin = data["rleft"],
                     Rmax = data["rleft"] + data["rdim"],
                     Zmin = data["zmid"] - 0.5*data["zdim"],
                     Zmax = data["zmid"] + 0.5*data["zdim"],
                     nx=data["nx"], ny=data["ny"]         # Number of grid points
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
        if hasattr(psinorm, "shape"):
            return reshape(p_spl(ravel(psinorm)),psinorm.shape)
        return p_spl(psinorm)

    def f_func(psinorm):
        if hasattr(psinorm, "shape"):
            return reshape(f_spl(ravel(psinorm)),psinorm.shape)
        return f_spl(psinorm)

    def pprime_func(psinorm):
        if hasattr(psinorm, "shape"):
            return reshape(pprime_spl(ravel(psinorm)),psinorm.shape)
        return pprime_spl(psinorm)
    
    def ffprime_func(psinorm):
        if hasattr(psinorm, "shape"):
            return reshape(ffprime_spl(ravel(psinorm)),psinorm.shape)
        return ffprime_spl(psinorm)

    
    # Create a set of profiles to calculate toroidal current density Jtor
    profiles = jtor.ProfilesPprimeFfprime(pprime_func,
                                          ffprime_func,
                                          data["rcentr"] * data["bcentr"],
                                          p_func=p_func, 
                                          f_func=f_func)

    # Use these profiles to calculate plasma psi
    # This requires a bit of a hack to set the poloidal flux
    
    coil_psi = machine.psi(eq.R, eq.Z)
    eq._updatePlasmaPsi(data["psi"] - coil_psi)
    
    # Perform a linear solve, calculating plasma psi
    eq.solve(profiles)

    print("Plasma current: {0} Amps, input: {1} Amps".format(eq.plasmaCurrent(), data["cpasma"]))
    
    # Identify points to constrain: X-points, O-points and separatrix shape

    psi_in = interpolate.RectBivariateSpline(eq.R[:,0], eq.Z[0,:], data["psi"])
    
    # Find all the O- and X-points
    opoint, xpoint = critical.find_critical(eq.R, eq.Z, data["psi"])
    
    if show:
        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)
        axis.set_aspect('equal')
        axis.set_xlabel("Major radius [m]")
        axis.set_ylabel("Height [m]")
    
        levels = linspace(amin(data["psi"]), amax(data["psi"]), 50)
        axis.contour(eq.R,eq.Z,data["psi"], levels=levels, colors='k')
        
        # Put red dots on X-points
        for r,z,_ in xpoint:
            axis.plot(r,z,'ro')

        sep_contour=axis.contour(eq.R, eq.Z, data["psi"], levels=[xpoint[0][2]], colors='r')
        
    # Find the separatrix
    isoflux = critical.find_separatrix(eq, opoint, xpoint, ntheta=ntheta, psi=data["psi"], axis=axis)

    # Find best fit for coil currents
    controlsystem = control.constrain(xpoints=xpoint, isoflux=isoflux, gamma=1e-14)
    controlsystem(eq)
    
    if show:
        axis.contour(eq.R,eq.Z, eq.psi(), levels=levels, colors='r')
        plt.pause(1)
    
    # Print the coil currents
    machine.printCurrents()

    ####################################################################
    # Refine the equilibrium to ensure consistency 
    # Solve using Picard iteration
    #
    
    picard.solve(eq,          # The equilibrium to adjust
                 profiles,    # The toroidal current profile function
                 controlsystem, show=show, axis=axis,
                 rtol=rtol)
    
    print("Plasma current: {0} Amps, input: {1} Amps".format(eq.plasmaCurrent(), data["cpasma"]))
    print("Plasma pressure on axis: {0} Pascals".format(eq.pressure(0.0)))
    machine.printCurrents()

    # Save the control system to eq
    eq.control = controlsystem
    
    return eq
