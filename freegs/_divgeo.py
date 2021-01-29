"""
Low level routines for reading and writing DivGeo files

Copyright 2018 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk

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
import numpy as np

from ._fileutils import f2s, ChunkOutput, write_1d, write_2d, next_value


def write(data, fh, label=None):
    """
    Write a DivGeo file, given a dictionary of data

    data - dictionary
      nx, ny        Number of points in R (x), Z (y)
      rcentr        Reference value of R
      bcentr        Vacuum toroidal magnetic field at rcentr
      sibdry        Poloidal flux psi at plasma boundary

      either:
        r[nx]       1D array of R values
      or
        rdim        Size of the R dimension
        rleft       Innermost R point

      either:
        z[ny]       1D array of Z values
      or
        zdim        Size of the Z dimension
        zmid        Middle of the Z dimension

    """

    # Write header
    fh.write(
        """    jm   :=  no. of grid points in radial direction;
    km   :=  no. of grid points in vertical direction;
    r    :=  radial   coordinates of grid points  [m];
    z    :=  vertical coordinates of grid points  [m];
    psi  :=  flux per radiant at grid points     [Wb/rad];
    psib :=  psi at plasma boundary              [Wb/rad];
    btf  :=  toroidal magnetic field                  [t];
    rtf  :=  major radius at which btf is specified   [m];
 
 
    jm    =          {nx:d} ;
    km    =          {ny:d} ;
    psib  =   {sibdry:1.15E}  Wb/rad;
    btf   =   {bcentr:1.14f}       t;
    rtf   =   {rcentr:1.14f}       m;
 
""".format(
            **data
        )
    )

    try:
        r = data["r"]
    except KeyError:
        # No "r" in the dictionary
        # use rdim, rleft and nx (from eqdsk)

        Rmin = data["rleft"]
        Rmax = data["rleft"] + data["rdim"]
        nx = data["nx"]

        dR = (Rmax - Rmin) / (nx - 1)
        r = np.arange(nx) * dR + Rmin

    try:
        z = data["z"]
    except KeyError:
        # No "z" in the dictionary
        # use zdim, zmid and ny (from eqdsk)

        Zmin = data["zmid"] - 0.5 * data["zdim"]
        Zmax = data["zmid"] + 0.5 * data["zdim"]
        ny = data["ny"]

        dZ = (Zmax - Zmin) / (ny - 1)
        z = np.arange(ny) * dZ + Zmin

    # Now write r and z
    fh.write("    r(1:jm);\n")
    co = ChunkOutput(fh, chunksize=5, extraspaces=3)
    write_1d(r, co)
    co.newline()

    fh.write(" \n    z(1:km);\n")
    write_1d(z, co)
    co.newline()

    fh.write(" \n      ((psi(j,k)-psib,j=1,jm),k=1,km)\n")

    write_2d(data["psi"] - data["sibdry"], co)
    co.newline()
