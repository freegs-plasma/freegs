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
        
    for coil in coils:
        axis.plot(coil["R"], coil["Z"], "sk")

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
