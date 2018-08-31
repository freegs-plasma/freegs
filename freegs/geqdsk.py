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
from .machine import Wall
from . import jtor
from . import control
from . import picard
from .gradshafranov import mu0

from scipy import interpolate
from numpy import linspace, amin, amax, reshape, ravel, zeros, argmax, clip, sin, cos, pi
import math
import numpy as np

from scipy.integrate import romb

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
    data['ffprime'] = eq.ffprime(psinorm)
    data['pprime'] = eq.pprime(psinorm)

    data["psi"] = psi
    
    qpsi = zeros([nx])
    qpsi[1:] = eq.q(psinorm[1:]) # Exclude axis
    qpsi[0] = qpsi[1]
    data["qpsi"] = qpsi
    
    if eq.tokamak.wall:
        data["rlim"] = eq.tokamak.wall.R
        data["zlim"] = eq.tokamak.wall.Z
    
    # Call fileformat to write the data
    fileformat(data, fh, label=label)

import matplotlib.pyplot as plt

def isPow2(val):
    """
    Returns True if val is a power of 2
    val     - Integer
    """
    return val & (val-1) == 0

def ceilPow2(val):
    """
    Find a power of two greater than or equal to val
    """
    return 2**math.ceil(math.log2(val))

def read(fh, machine, rtol=1e-3, ntheta=8, show=False, axis=None, cocos=1, domain=None, blend=0.0, fit_sol=False):
    """
    Reads a G-EQDSK format file
    
    fh : File handle
    machine : Machine object defining coil locations
    rtol : float
        Relative error in nonlinear solve
    ntheta : integer
        Number of points in poloidal angle on the separatrix
        this is used to constrain the plasma shape
    show : Boolean
        Set to true to show solution in a new figure
    axis : Matplotlib Axis object
       Set to an axis for plotting. Implies show=True
    cocos : integer
        COordinate COnventions. Not fully handled yet,
        only whether psi is divided by 2pi or not.
        if < 10 then psi is divided by 2pi, otherwise not.
    domain : list/tuple of 4 elements
        Sets the (R,Z) domain to solve for
        (Rmin, Rmax, Zmin, Zmax)
    blend : float between 0 and 1
        Weighting of the previous poloidal flux at each step of the
        Picard iteration. The default (0.0) is to use no blending.
        Blending slows convergence, but can stabilise some oscillating
        unstable solutions.
    fit_sol : Boolean
        If False (default) then only the poloidal flux inside the
        separatrix is used to constrain the coil currents.
        This is particularly for reading SCENE input, which is not valid 
        outside the separatrix.
        If True, the whole domain is used in the fitting.
        This is useful if the locations of strike points need to be constrained.

    A nonlinear solve will be performed, using Picard iteration
    
    Returns
    -------

    An Equilibrium object eq. In addition, the following is available:

    eq.control   - The control system
    eq._profiles - The profiles object
    
    """

    if fit_sol and domain:
        raise ValueError("Sorry, fit_sol cannot be used with the domain keyword")
    
    if axis is not None:
        show = True

    # Read the data as a Dictionary
    data = _geqdsk.read(fh, cocos=cocos)
    
    # If data contains a limiter, set the machine wall
    if "rlim" in data:
        if len(data["rlim"]) > 3:
            machine.wall = Wall(data["rlim"], data["zlim"])
        else:
            print("Fewer than 3 points given for limiter/wall. Ignoring.")
    

    nx = data["nx"]
    ny = data["ny"]
    psi = data["psi"]
    
    if not (isPow2(nx-1) and isPow2(ny-1)):
        print("Warning: Input grid size %d x %d has sizes which are not 2^n+1" % (nx, ny))
        
        rin = linspace(0, 1, nx)
        zin = linspace(0, 1, ny)
        psi_interp = interpolate.RectBivariateSpline(rin, zin, psi, kx=1, ky=1)
        
        # Ensure that newnx, newny is 2^n + 1
        nx = ceilPow2(nx-1) + 1
        ny = ceilPow2(ny-1) + 1
        print("   => Resizing to %d x %d" % (nx, ny))

        rnew = linspace(0, 1, nx)
        znew = linspace(0, 1, ny)

        # Interpolate onto new grid
        psi = psi_interp(rnew, znew)

        
    # Range of psi normalises psi derivatives
    psi_bndry = data["sibdry"]
    psi_axis = data["simagx"]
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
    
    # Calculate normalised psi.
    # 0 = magnetic axis
    # 1 = plasma boundary
    psi_norm = clip((psi - psi_axis)  / (psi_bndry - psi_axis), 0.0, 1.0)

    # Create masking function: 1 inside plasma, 0 outside
    mask = np.ones(psi.shape)
    mask[psi_norm > 1. - 1e-6] = 0.0  # Ignore areas outside the plasma
    
    # Create an Equilibrium object
    eq = Equilibrium(tokamak = machine,
                     Rmin = data["rleft"],
                     Rmax = data["rleft"] + data["rdim"],
                     Zmin = data["zmid"] - 0.5*data["zdim"],
                     Zmax = data["zmid"] + 0.5*data["zdim"],
                     nx=nx, ny=ny         # Number of grid points
                 )
    # Grid spacing
    dR = eq.R[1,0] - eq.R[0,0]
    dZ = eq.Z[0,1] - eq.Z[0,0]
    
    # Toroidal current
    Jtor = eq.R * pprime_func(psi_norm) + ffprime_func(psi_norm)/(eq.R * mu0)
    Jtor *= mask
    
    # Quick calculation of total toroidal current
    print("CURRENT: ", romb(romb(Jtor)) * dR*dZ)
    
    if domain:
        # Change the (R,Z) domain, increasing the resolution if needed
        # to keep the resolution as good or better than the input
        Rmin, Rmax, Zmin, Zmax = domain
        # Calculate approximate grid size needed to keep current resolution
        fnewnx = (Rmax - Rmin)/dR + 1.0
        if fnewnx > nx:
            newnx = int(ceilPow2(fnewnx-1))+1
        fnewny = (Zmax - Zmin)/dZ + 1.0
        if fnewny > ny:
            newny = int(ceilPow2(fnewny-1))+1

        if (newnx != nx) or (newny != ny):
            print("Changing resolution: {} x {}".format(newnx,newny))
            
        # Create an interpolation function for Jtor and the input psi
        Jtor_func = interpolate.RectBivariateSpline(eq.R[:,0], eq.Z[0,:], Jtor)
        psi_func = interpolate.RectBivariateSpline(eq.R[:,0], eq.Z[0,:], psi)
        
        # Create a new Equilibrium object
        # (replacing previous 'eq')
        eq = Equilibrium(tokamak = machine,
                         Rmin = Rmin,
                         Rmax = Rmax,
                         Zmin = Zmin,
                         Zmax = Zmax,
                         nx=nx, ny=ny
                         )
        
        # Interpolate Jtor and psi onto new grid
        Jtor = Jtor_func(eq.R, eq.Z, grid=False)
        psi = psi_func(eq.R, eq.Z, grid=False)

        # Update the mask function by calculating normalised psi
        # on the new grid
        psi_norm = clip((psi - psi_axis)  / (psi_bndry - psi_axis), 0.0, 1.0)

        # Create masking function: 1 inside plasma, 0 outside
        mask = np.ones(psi.shape)
        mask[psi_norm > 1. - 1e-6] = 0.0  # Ignore areas outside the plasma

    # Note: Here we have
    #   eq : Equilibrium object
    #   Jtor : 2D array (nx,ny) Toroidal current density
    #   psi  : 2D array (nx,ny) Input poloidal flux
    #   psi_norm : 2D array (nx,ny) Normalised input poloidal flux
    #   mask : 2D array (nx,ny) 1 inside plasma, 0 outside

    # Perform a linear solve to calculate psi
    # using known Jtor
    eq.solve(profiles, Jtor=Jtor)
    
    print("Plasma current: {0} Amps, input: {1} Amps".format(eq.plasmaCurrent(), data["cpasma"]))
    
    # Identify points to constrain: X-points, O-points and separatrix shape

    if show:
        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)
        axis.set_aspect('equal')
        axis.set_xlabel("Major radius [m]")
        axis.set_ylabel("Height [m]")

        # Note: psi can be offset by a constant in the input
        axis.contour(eq.R, eq.Z, psi, 50, colors='k')

        # Find all the O- and X-points
        opoint, xpoint = critical.find_critical(eq.R, eq.Z, psi)        
        
        # Put red dots on X-points
        for r,z,_ in xpoint:
            axis.plot(r,z,'ro')

        # Draw separatrix if there is an X-point
        if len(xpoint) > 0:
            sep_contour=axis.contour(eq.R, eq.Z, psi, levels=[xpoint[0][2]], colors='r')
        
    # Find best fit for coil currents
    # First create a control system (see control.py)
    if fit_sol:
        controlsystem = control.ConstrainPsi2D(psi) # Fit entire domain
    else:
        controlsystem = control.ConstrainPsi2D(psi, weights=mask) # Remove SOL from fitting
    # Run control system to find coil currents
    controlsystem(eq)
    
    if show:
        axis.contour(eq.R,eq.Z, eq.psi(), 50, colors='r')
        plt.pause(1)
    
    # Print the coil currents
    machine.printCurrents()

    ####################################################################
    # Refine the equilibrium to ensure consistency 
    # Solve using Picard iteration
    #

    controlsystem = control.ConstrainPsiNorm2D(psi_norm, weights=mask)
    
    picard.solve(eq,          # The equilibrium to adjust
                 profiles,    # The toroidal current profile function
                 controlsystem, show=show, axis=axis,
                 rtol=rtol, blend=0.5)
    
    print("Plasma current: {0} Amps, input: {1} Amps".format(eq.plasmaCurrent(), data["cpasma"]))
    print("Plasma pressure on axis: {0} Pascals".format(eq.pressure(0.0)))
    machine.printCurrents()

    # Attempt to find O- and X-points
    psi = eq.psi()
    opoint, xpoint = critical.find_critical(eq.R, eq.Z, psi)

    if xpoint:
        # Use x-point and o-point constraints because the size of the grid may be changed
        # in which case the 2D psi constraints would fail
        
        # Find the separatrix
        isoflux = critical.find_separatrix(eq, opoint, xpoint, ntheta=ntheta, psi=psi, axis=axis)
        
        # Save the control system to eq
        eq.control = control.constrain(xpoints=xpoint, isoflux=isoflux, gamma=1e-14)
        
    return eq
