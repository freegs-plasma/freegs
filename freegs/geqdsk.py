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

def read(fh, machine, rtol=1e-3, ntheta=8, show=False):
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
    
    eq.solve(profiles, niter=10, sublevels=5, ncycle=5)

    print("Plasma current: {0} Amps, input: {1} Amps".format(eq.plasmaCurrent(), data["cpasma"]))
    
    # Identify points to constrain: X-points, O-points

    psi_in = interpolate.RectBivariateSpline(eq.R[:,0], eq.Z[0,:], data["psi"])
    
    opoint, xpoint = critical.find_critical(eq.R, eq.Z, data["psi"])
    
    axis = None
    if show:
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

    isoflux = []
    
    def find_separatrix(r0,z0, r1,z1, n=100):
        """
        (r0,z0) - Start location inside separatrix
        (r1,z1) - Location outside separatrix
        
        n - Number of starting points to use
        """
        # Clip (r1,z1) to be inside domain
        # Shorten the line so that the direction is unchanged
        if abs(r1 - r0) > 1e-6:
            rclip = clip(r1, eq.Rmin, eq.Rmax)
            z1 = z0 + (z1 - z0) * abs( (rclip - r0) / (r1 - r0) )
            r1 = rclip
        
        if abs(z1 - z0) > 1e-6:
            zclip = clip(z1, eq.Zmin, eq.Zmax)
            r1 = r0 + (r1 - r0) * abs( (zclip - z0) / (z1 - z0) )
            z1 = zclip

        r = linspace(r0, r1, n)
        z = linspace(z0, z1, n)
        
        if show:
            axis.plot(r,z)
        
        pnorm = (psi_in(r, z, grid=False) - opoint[0][2])/(xpoint[0][2] - opoint[0][2])
        ind = argmax(pnorm>1.0)

        f = (pnorm[ind] - 1.0)/(pnorm[ind] - pnorm[ind-1])
        
        r = (1. - f) * r[ind] + f * r[ind-1]
        z = (1. - f) * z[ind] + f * z[ind-1]
        
        if show:
            axis.plot(r,z,'bo')
        
        return r,z

    # Find points on the separatrix to constrain plasma shape
    if ntheta > 0:
        for theta in linspace(0, 2*pi, ntheta, endpoint=False):
            r0, z0 = opoint[0][0:2]
            r,z = find_separatrix(r0, z0, r0 + 10.*sin(theta), z0 + 10.*cos(theta))
            isoflux.append( (r,z, xpoint[0][0], xpoint[0][1]) )
    
    # Find best fit for coil currents
    controlsystem = control.constrain(xpoints=xpoint, isoflux=isoflux, gamma=1e-14)
    controlsystem(eq)
    
    psi = eq.psi()
    
    if show:
        axis.contour(eq.R,eq.Z, psi, levels=levels, colors='r')
        plt.pause(1)
        
    machine.printCurrents()

    ####################################################################
    # Refine the equilibrium to ensure consistency 
    # Solve using Picard iteration
    #
    
    picard.solve(eq,          # The equilibrium to adjust
                 profiles,    # The toroidal current profile function
                 controlsystem, show=show, axis=axis,
                 niter=20, sublevels=5, ncycle=20, rtol=rtol)
    
    print("Plasma current: {0} Amps, input: {1} Amps".format(eq.plasmaCurrent(), data["cpasma"]))
    print("Plasma pressure on axis: {0} Pascals".format(eq.pressure(0.0)))
    machine.printCurrents()

    return eq
