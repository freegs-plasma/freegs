"""
Low level routines for reading and writing G-EQDSK files

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

from datetime import date
from numpy import zeros, linspace
import re

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

    def newline(self):
        """
        Ensure that the file is at the start of a new line
        """
        if self.counter != 0:
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


def write(data, fh, label=None):
    """
    Write a GEQDSK equilibrium file, given a dictionary of data
    
    data - dictionary
      nx, ny        Number of points in R (x), Z (y)
      rdim, zdim    Sizes of the R,Z dimensions
      rcentr        Reference value of R
      bcentr        Vacuum toroidal magnetic field at rcentr
      rleft         R at left (inner) boundary
      zmid          Z at middle of domain
      rmagx, zmagx  R,Z at magnetic axis (O-point)
      simagx        Poloidal flux psi at magnetic axis
      sibdry        Poloidal flux psi at plasma boundary
      cpasma        Plasma current [Amps]   

      fpol          1D array of f(psi)=R*Bt  [meter-Tesla]
      pres          1D array of p(psi) [Pascals]
      qpsi          1D array of q(psi)
      
      psi           2D array (nx,ny) of poloidal flux
    
    fh - file handle
    
    label - Text label to put in the file
    """
    
    nx = data["nx"]
    ny = data["ny"]
    
    if not label:
        label = "FREEGS %s" % date.today().strftime("%d/%m/%Y")
        
    # First line: Identification string, followed by resolution
    fh.write("  " + label + "   3  {0}  {1}\n".format(nx, ny))
    
    # Second line
    fh.write(f2s(data["rdim"])+f2s(data["zdim"])+f2s(data["rcentr"])+f2s(data["rleft"])+f2s(data["zmid"])+"\n")
    
    # Third line
    fh.write(f2s(data["rmagx"]) + f2s(data["zmagx"]) + f2s(data["simagx"]) + f2s(data["sibdry"]) + f2s(data["bcentr"]) + "\n")
    
    # 4th line
    fh.write(f2s(data["cpasma"]) + f2s(data["simagx"]) + f2s(0.0) + f2s(data["rmagx"]) + f2s(0.0) + "\n")
    
    # 5th line
    fh.write(f2s(data["zmagx"]) + f2s(0.0) + f2s(data["sibdry"]) + f2s(0.0) + f2s(0.0) + "\n")

    # fill arrays
    workk = zeros([nx])
    
    # Write arrays
    co = ChunkOutput(fh)

    write_1d(data["fpol"], co)
    write_1d(data["pres"], co)
    write_1d(workk, co)
    write_1d(workk, co)
    write_2d(data["psi"], co)
    write_1d(data["qpsi"], co)

    # Boundary / limiters
        
    nbdry = 0
    nlim = 0
    if "rbdry" in data:
        nbdry = len(data["rbdry"])
    if "rlim" in data:
        nlim = len(data["rlim"])
    
    co.newline()
    fh.write("   {0}   {1}\n".format(nbdry, nlim))

    if nbdry > 0:
        for r,z in zip(data["rbdry"], data["zbdry"]):
            co.write(r)
            co.write(z)
        co.newline()
    
    if nlim > 0:
        for r,z in zip(data["rlim"], data["zlim"]):
            co.write(r)
            co.write(z)
        co.newline()
    
def next_value(fh):
    """
    A generator which yields values from a file handle
    
    Checks if the value is a float or int, returning
    the correct type depending on if '.' is in the string
    
    """
    pattern = re.compile(r'[ \-]\d(?:\.\d+[Ee][\+\-]\d\d)?')

    # Go through each line, extract values, then yield them one by one
    for line in fh:
        matches = pattern.findall(line)
        for match in matches:
            if '.' in match:
                yield float(match)
            else:
                yield int(match)
    
def read(fh):
    """
    Read a G-EQDSK formatted equilibrium file
    
    Format is specified here:
    https://fusion.gat.com/theory/Efitgeqdsk

    Returns
    -------
    
    A dictionary containing:
      nx, ny        Number of points in R (x), Z (y)
      rdim, zdim    Sizes of the R,Z dimensions
      rcentr        Reference value of R
      bcentr        Vacuum toroidal magnetic field at rcentr
      rleft         R at left (inner) boundary
      zmid          Z at middle of domain
      rmagx, zmagx  R,Z at magnetic axis (O-point)
      simagx        Poloidal flux psi at magnetic axis
      sibdry        Poloidal flux psi at plasma boundary
      cpasma        Plasma current [Amps]   

      fpol          1D array of f(psi)=R*Bt  [meter-Tesla]
      pres          1D array of p(psi) [Pascals]
      qpsi          1D array of q(psi)
      
      psi           2D array (nx,ny) of poloidal flux
    
    """

    # Read the first line
    header = fh.readline()
    words = header.split()  # Split on whitespace
    if len(words) < 3:
        raise ValueError("Expecting at least 3 numbers on first line")

    idum = int(words[-3])
    nx = int(words[-2])
    ny = int(words[-1])

    print("  nx = {0}, ny = {1}".format(nx, ny))

    # Dictionary to hold result
    data = {"nx":nx, "ny":ny}

    # List of fields to read. None means discard value
    fields = ["rdim",  "zdim",  "rcentr", "rleft", "zmid",
              "rmagx", "zmagx", "simagx", "sibdry", "bcentr",
              "cpasma","simagx", None,    "rmagx",  None,
              "zmagx",  None,   "sibdry",  None,    None]

    values = next_value(fh)
    
    for f in fields:
        val = next(values)
        if f:
            data[f] = val

    # Read arrays

    def read_1d(n):
        """
        Read a 1D array of length n
        """
        val = zeros(n)
        for i in range(n):
            val[i] = next(values)
        return val

    def read_2d(n,m):
        """
        Read a 2D (n,m) array in Fortran order
        """
        val = zeros([n,m])
        for y in range(m):
            for x in range(n):
                val[x,y] = next(values)
        return val
    
    data["fpol"] = read_1d(nx)
    data["pres"] = read_1d(nx)
    data["workk1"] = read_1d(nx)
    data["workk2"] = read_1d(nx)
    
    data["psi"] = read_2d(nx,ny)
    
    data["qpsi"] = read_1d(nx)
    
    nbdry = next(values)
    nlim = next(values)
    
    if nbdry > 0:
        # Read (R,Z) pairs
        data["rbdry"] = zeros(nbdry)
        data["zbdry"] = zeros(nbdry)
        for i in range(nbdry):
            data["rbdry"][i] = next(values)
            data["zbdry"][i] = next(values)
            
    if nlim > 0:
        # Read (R,Z) pairs
        data["rlim"] = zeros(nbdry)
        data["zlim"] = zeros(nbdry)
        for i in range(nbdry):
            data["rlim"][i] = next(values)
            data["zlim"][i] = next(values)
    
    return data
