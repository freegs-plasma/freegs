import matplotlib.pyplot as plt

from numpy import linspace, amin, amax

def plotEquilibrium(R, Z, psi, axis=None):
    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot(111)

    levels = linspace(amin(psi), amax(psi), 100)
    
    axis.contour(R,Z,psi, levels=levels)
    axis.set_aspect('equal')
    axis.set_xlabel("Major radius [m]")
    axis.set_ylabel("Height [m]")

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
