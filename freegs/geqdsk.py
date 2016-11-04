from datetime import date
from numpy import zeros, linspace

from . import critical

def f2s(f):
    """
    Format a string containing a float
    """
    s = ""
    if f >= 0.0:
        s += " "
    return s + "%1.9E" % f


class ChunkOutput:
    """
    This outputs values in lines, inserting
    newlines when needed.
    """
    def __init__(self, filehandle, chunksize=5):
        """
        filehandle - output to write to
        chunksize - number of values on a line
        """
        self.fh = filehandle
        self.counter = 0
        self.chunk = chunksize

    def write(self, value):
        """"
        Write a value to the output, adding a newline if needed
        """
        self.fh.write(f2s(value))
        self.counter += 1
        if self.counter == self.chunk:
            self.fh.write("\n")
            self.counter = 0

def write_1d(val, out):
    """
    Writes a 1D variable val to the file handle out
    """
    for i in range(len(val)):
        out.write(val[i])

def write_2d(val, out):
    """
    Writes a 2D array. Note that this transposes
    the array, looping over the first index fastest
    """
    nx,ny = val.shape
    for y in range(ny):
        for x in range(nx):
            out.write(val[x,y])


def write(eq, fh, label=None, oxpoints=None):
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

    if not label:
        label = "FREEGS %s" % date.today().strftime("%d/%m/%Y")
    # First line: Identification string, followed by resolution
    fh.write("  " + label + "   3  {0}  {1}\n".format(nx, ny))
    
    rmin = eq.Rmin
    rmax = eq.Rmax
    zmin = eq.Zmin
    zmax = eq.Zmax
    
    fvac = eq.fpolVac() # Vacuum f = R*Bt
    R0 = 1.0 # Reference location
    B0 = fvac / R0 # Reference vacuum toroidal magnetic field

    # Second line
    rdim = rmax - rmin # Horizontal dimension in meter of computational box
    zdim = zmax - zmin # Vertical dimension in meter of computational box
    rcentr = R0   # R in meter of vacuum toroidal magnetic field BCENTR
    rleft = rmin  # Minimum R in meter of rectangular computational box
    zmid = 0.5*(zmin + zmax) # Z of center of computational box in meter

    fh.write(f2s(rdim)+f2s(zdim)+f2s(rcentr)+f2s(rleft)+f2s(zmid)+"\n")
    
    # Third line
    
    rmaxis, zmaxis, simag =  opoint[0] # agnetic axis
    sibdry = xpoint[0][2]  # Psi at boundary
    bcentr = B0 # Vacuum magnetic field at rcentr

    fh.write(f2s(rmaxis) + f2s(zmaxis) + f2s(simag) + f2s(sibdry) + f2s(bcentr) + "\n")
    
    # 4th line
    
    cpasma = eq.plasmaCurrent()  # Plasma current
    fh.write(f2s(cpasma) + f2s(simag) + f2s(0.0) + f2s(rmaxis) + f2s(0.0) + "\n")
    
    # 5th line
    
    fh.write(f2s(zmaxis) + f2s(0.0) + f2s(sibdry) + f2s(0.0) + f2s(0.0) + "\n")

    # fill arrays
    workk = zeros([nx])
    
    psinorm = linspace(0.0, 1.0, nx, endpoint=False) # Does not include separatrix
    fpol = eq.fpolPsiN(psinorm)
    pres = eq.pressurePsiN(psinorm)
    
    qpsi = zeros([nx])
    qpsi[1:] = eq.qPsiN(psinorm[1:]) # Exclude axis
    qpsi[0] = qpsi[1]
    
    # Write arrays
    co = ChunkOutput(fh)
    write_1d(fpol, co)
    write_1d(pres, co)
    write_1d(workk, co)
    write_1d(workk, co)
    write_2d(psi, co)
    write_1d(qpsi, co)

    # Boundary / limiters
    
    if co.counter != 0:
        fh.write("\n")
    fh.write("   0   0\n")
