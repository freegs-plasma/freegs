"""
Plotting using matplotlib

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

import matplotlib.pyplot as plt

from numpy import linspace, amin, amax
from . import critical

def plotEquilibrium(eq, axis=None, show=True, oxpoints=True):
    """
    Plot the equilibrium flux surfaces

    axis - Specify the axis on which to plot
    
    """

    R = eq.R
    Z = eq.Z
    psi = eq.psi()

    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot(111)

    levels = linspace(amin(psi), amax(psi), 100)
    
    axis.contour(R,Z,psi, levels=levels)
    axis.set_aspect('equal')
    axis.set_xlabel("Major radius [m]")
    axis.set_ylabel("Height [m]")

    if oxpoints:
        # Add O- and X-points
        opt, xpt = critical.find_critical(eq.R, eq.Z, psi)
        
        for r,z,_ in xpt:
            axis.plot(r,z,'ro')
        for r,z,_ in opt:
            axis.plot(r,z,'go')
            
        if xpt:
            psi_bndry = xpt[0][2]
            sep_contour=axis.contour(eq.R, eq.Z,psi, levels=[psi_bndry], colors='r')
        
    if show:
        plt.show()
    
    return axis

def plotCoils(coils, axis=None):
    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot(111)
     
        
    return axis


def plotConstraints(xpoints, axis=None):
    """
    Plots constraints used for coil currents
    
    """

    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot(111)

    for xpt in xpoints:
        axis.plot(xpt[0], xpt[1], 'ro')    

    return axis
