"""
An interface class to the g-eqdsk file format, using the fortranformat library to directly
implement Fortran format reads and writes.
"""
from __future__ import annotations
from fortranformat import FortranRecordReader as FReader
from fortranformat import FortranRecordWriter as FWriter
from datetime import date
from warnings import warn
from pathlib import Path
import numpy as np
from scipy import interpolate
import freegs


class GEqdskFile: 
    """
    A file-type for saving/reading equilibria from file.
    Using the definition of the g-eqdsk format from https://w3.pppl.gov/ntcc/TORAY/G_EQDSK.pdf

    A right-handed cylindrical coordinate system (R, φ, Z) is used. The G EQDSK provides
    information on the pressure, poloidal current function, q profile on a uniform flux
    grid from the magnetic axis to the plasma boundary and the poloidal flux
    function on the rectangular computation grid. Information on the plasma
    boundary and the surrounding limiter contour in also provided.

    CASE    : Identification character string
    NW      : Number of horizontal R grid points
    NH      : Number of vertical Z grid points
    RDIM    : Horizontal dimension in meter of computational box
    ZDIM    : Vertical dimension in meter of computational box
    RLEFT   : Minimum R in meter of rectangular computational box
    ZMID    : Z of center of computational box in meter
    RMAXIS  : R of magnetic axis in meter
    ZMAXIS  : Z of magnetic axis in meter
    SIMAG   : poloidal flux at magnetic axis in Weber /rad
    SIBRY   : poloidal flux at the plasma boundary in Weber /rad
    RCENTR  : R in meter of vacuum toroidal magnetic field BCENTR
    BCENTR  : Vacuum toroidal magnetic field in Tesla at RCENTR
    CURRENT : Plasma current in Ampere
    FPOL    : Poloidal current function in m-T, F = RBT on uniform flux grid
    PRES    : Plasma pressure in nt / m2 on uniform flux grid
    FFPRIM  : FF'(ψ) in (mT)2 / (Weber /rad) on uniform flux grid
    PPRIME  : P'(ψ) in (nt /m2) / (Weber /rad) on uniform flux grid
    PSIRZ   : Poloidal flux in Weber / rad on the rectangular grid points
    QPSI    : q values on uniform flux grid from axis to boundary
    NBBBS   : Number of boundary points
    LIMITR  : Number of limiter points
    RBBBS   : R of boundary points in meter
    ZBBBS   : Z of boundary points in meter
    RLIM    : R of surrounding limiter contour in meter
    ZLIM    : Z of surrounding limiter contour in meter
    N.b. The toroidal current JT related to P'(ψ) and FF'(ψ) through
    JT (Amp/m2) = R P'(ψ) + FF'(ψ) / R
    """
    @classmethod
    def from_file(cls, file: Path, fortran_order_arrays: bool=True) -> GEqdskFile:
        """
        Read a g-eqdsk file and return a GEqdskFile object
        """
        self = cls()
        self.read(file, fortran_order_arrays=fortran_order_arrays)
        return self

    def __init__(self): 
        """
        Make a blank GEqdskFile object. Variables are initialised with their
        expected type to help with type-hinting.
        """
        self.CASE    = str()
        self.NW      = int()
        self.NH      = int()
        self.RDIM    = float()
        self.ZDIM    = float()
        self.RLEFT   = float()
        self.ZMID    = float()
        self.RMAXIS  = float()
        self.ZMAXIS  = float()
        self.SIMAG   = float()
        self.SIBRY   = float()
        self.RCENTR  = float()
        self.BCENTR  = float()
        self.CURRENT = float()
        self.FPOL    = np.zeros((0,))
        self.PRES    = np.zeros((0,))
        self.FFPRIM  = np.zeros((0,))
        self.PPRIME  = np.zeros((0,))
        self.PSIRZ   = np.zeros((0,0))
        self.QPSI    = np.zeros((0,))
        self.NBBBS   = np.zeros((0,))
        self.LIMITR  = np.zeros((0,))
        self.RBBBS   = np.zeros((0,))
        self.ZBBBS   = np.zeros((0,))
        self.RLIM    = np.zeros((0,))
        self.ZLIM    = np.zeros((0,))

    def read(self, file: Path, fortran_order_arrays: bool=True):
        """
        Reads a g-eqdsk file according the specification in https://w3.pppl.gov/ntcc/TORAY/G_EQDSK.pdf
        The 'fortranformat' library is used to directly use the fortran format specifications
        """
        if isinstance(file, str):
            file = Path(file)

        # Read the eqdsk file as text, and then split into a list of lines
        f = file.read_text().split("\n")

        # read (neqdsk,2000) (case(i),i=1,6),idum,nw,nh
        self.CASE, _, _, _, _, _, _, self.NW, self.NH = FReader("(6a8,3i4)").read(f[0])
        # read (neqdsk,'5e16.9') rdim,zdim,rcentr,rleft,zmid
        self.RDIM, self.ZDIM, self.RCENTR, self.RLEFT, self.ZMID = FReader("(5e16.9)").read(f[1])
        # read (neqdsk,'5e16.9') rmaxis,zmaxis,simag,sibry,bcentr
        self.RMAXIS, self.ZMAXIS, self.SIMAG, self.SIBRY, self.BCENTR = FReader("(5e16.9)").read(f[2])
        # read (neqdsk,'5e16.9') current,simag,xdum,rmaxis,xdum
        self.CURRENT, SIMAG_DUP, _, RMAXIS_DUP, _ = FReader("(5e16.9)").read(f[3])
        # read (neqdsk,'5e16.9') zmaxis,xdum,sibry,xdum,xdum
        ZMAXIS_DUP, _, SIBRY_DUP, _, _ = FReader("(5e16.9)").read(f[4])

        # Assert duplicate reads match (i.e. make sure that the file is consistent formatted)
        assert np.isclose(self.SIMAG, SIMAG_DUP)
        assert np.isclose(self.RMAXIS, RMAXIS_DUP)
        assert np.isclose(self.ZMAXIS, ZMAXIS_DUP)
        assert np.isclose(self.SIBRY, SIBRY_DUP)

        # Initialise the value reader, and starting reading values from line 5 onwards
        # IMPORTANT: do not reinitialise the reader, since otherwise we'll loose the position of the reader
        # Note that line_numbers will be reported from the read-offset
        value_reader = read_single_values(f, start_at_line=5)

        # read (neqdsk,'5e16.9') (fpol(i),i=1,nw)
        self.FPOL = read_1d_array(value_reader, number_of_values=self.NW)[0]
        # read (neqdsk,'5e16.9') (pres(i),i=1,nw)
        self.PRES = read_1d_array(value_reader, number_of_values=self.NW)[0]
        # read (neqdsk,'5e16.9') (ffprim(i),i=1,nw)
        self.FFPRIM = read_1d_array(value_reader, number_of_values=self.NW)[0]
        # read (neqdsk,'5e16.9') (pprime(i),i=1,nw)
        self.PPRIME = read_1d_array(value_reader, number_of_values=self.NW)[0]
        # read (neqdsk,'5e16.9') ((psirz(i,j),i=1,nw),j=1,nh)
        self.PSIRZ = read_2d_array(
            value_reader,
            rows=self.NH,
            columns=self.NW,
            fortran_order_arrays=fortran_order_arrays,
        )[0]
        # read (neqdsk,'5e16.9') (qpsi(i),i=1,nw)
        self.QPSI, line_number, _ = read_1d_array(value_reader, number_of_values=self.NW)
        # read (neqdsk,2022) nbbbs,limitr
        self.NBBBS, self.LIMITR = FReader("(2i5)").read(f[line_number + 1])

        # Make a new value reader, starting after the line with the integers
        value_reader = read_single_values(
            f, start_at_line=line_number + 2
        )
        # read (neqdsk,'5e16.9') (rbbbs(i),zbbbs(i),i=1,nbbbs)
        BBBS = read_1d_array(value_reader, number_of_values= 2 * self.NBBBS)[0]
        self.RBBBS, self.ZBBBS = BBBS[0::2], BBBS[1::2]
        # read (neqdsk,'5e16.9') (rlim(i),zlim(i),i=1,limitr)
        LIM = read_1d_array(value_reader, number_of_values=2 * self.LIMITR)[0]
        self.RLIM, self.ZLIM = LIM[0::2], LIM[1::2]

    def write(self, file: Path, comment: str="", fortran_order_arrays: bool=True): 
        """
        Writes a g-eqdsk file according the specification in https://w3.pppl.gov/ntcc/TORAY/G_EQDSK.pdf (for
        a GEqdskFile which has already been initialised)
        The 'fortranformat' library is used to directly use the fortran format specifications
        """

        if isinstance(file, str):
            file = Path(file)

        with open(file.absolute(), 'w') as f:
            # write (neqdsk,6a8,3i4) (case(i),i=1,6),idum,nw,nh
            # Here, we combine the space for the header into CASE, date, comment, 0, NW, NH
            CASE = self.CASE
            if len(CASE) > 8:
                warn(f"CASE must be 8 characters or less. {CASE} shortened to {CASE[0:8]}")
            if len(comment) > 28:
                warn(f"comment must be 28 characters of less. {comment} shortened to {comment[0:28]}")
            
            header_string = f" {date.today().strftime('%d/%m/%Y'):9s} {comment:28s}"
            f.write(FWriter('a8,a40,3i4').write([CASE, header_string, 0, self.NW, self.NH]) + '\n')
            # write (neqdsk,5e16.9) rdim,zdim,rcentr,rleft,zmid
            f.write(FWriter('5ES16.9').write([self.RDIM, self.ZDIM, self.RCENTR, self.RLEFT, self.ZMID]) + '\n')
            # write (neqdsk,'5e16.9') rmaxis,zmaxis,simag,sibry,bcentr
            f.write(FWriter('5ES16.9').write([self.RMAXIS, self.ZMAXIS, self.SIMAG, self.SIBRY, self.BCENTR]) + '\n')
            # write (neqdsk,'5e16.9') current,simag,xdum,rmaxis,xdum
            f.write(FWriter('5ES16.9').write([self.CURRENT, self.SIMAG, 0.0, self.RMAXIS, 0.0]) + '\n')
            # write (neqdsk,'5e16.9') zmaxis,xdum,sibry,xdum,xdum
            f.write(FWriter('5ES16.9').write([self.ZMAXIS, 0.0, self.SIBRY, 0.0, 0.0]) + '\n')

            # write (neqdsk,'5e16.9') (fpol(i),i=1,nw)
            f.write(write_1d_array(self.FPOL, 'ES16.9', max_line_length=5, array_length=self.NW))
            # write (neqdsk,'5e16.9') (pres(i),i=1,nw)
            f.write(write_1d_array(self.PRES, 'ES16.9', max_line_length=5, array_length=self.NW))
            # write (neqdsk,'5e16.9') (ffprim(i),i=1,nw)
            f.write(write_1d_array(self.FFPRIM, 'ES16.9', max_line_length=5, array_length=self.NW))
            # write (neqdsk,'5e16.9') (pprime(i),i=1,nw)
            f.write(write_1d_array(self.PPRIME, 'ES16.9', max_line_length=5, array_length=self.NW))
            # write (neqdsk,'5e16.9') ((psirz(i,j),i=1,nw),j=1,nh)
            f.write(
                write_1d_array(self.PSIRZ.flatten(order='F' if fortran_order_arrays else 'C'), 'ES16.9', max_line_length=5)
            )
            # write (neqdsk,'5e16.9') (qpsi(i),i=1,nw)
            f.write(write_1d_array(self.QPSI, 'ES16.9', max_line_length=5, array_length=self.NW))
            # write (neqdsk,2i5) nbbbs,limitr
            f.write(write_1d_array(np.array([self.NBBBS, self.LIMITR]), '2i5', max_line_length=5, array_length=2))
            # write (neqdsk,'5e16.9') (rbbbs(i),zbbbs(i),i=1,nbbbs)
            BBBS = np.zeros(2 * self.NBBBS)
            BBBS[0::2], BBBS[1::2] = self.RBBBS, self.ZBBBS
            f.write(write_1d_array(BBBS, 'ES16.9', max_line_length=5, array_length=2 * self.NBBBS))
            # write (neqdsk,'5e16.9') (rlim(i),zlim(i),i=1,limitr)
            LIM = np.zeros(2 * self.LIMITR)
            LIM[0::2], LIM[1::2] = self.RLIM, self.ZLIM
            f.write(write_1d_array(LIM, 'ES16.9', max_line_length=5, array_length=2 * self.LIMITR))
    
    @classmethod
    def from_equilibrium(cls, equilibrium: freegs.equilibrium.Equilibrium,
                         CASE: str="FREEGS", n_boundary_points: int=1001,
                         clockwise_current: bool=True, clockwise_B_tor: bool=True,
                         ) -> GEqdskFile:
        """
        Read a g-eqdsk file and return a GEqdskFile object

        clockwise_current and clockwise_B_tor define the direction of the plasma current and toroidal field,
        which together define the helicity of the plasma.
        Here, "clockwise" means in the clockwise direction if viewed from above, or (equivalently) in the
        R \cross Z direction.
        """
        self = cls()
        psi = equilibrium.psi()

        current_factor = -1.0 if clockwise_current else 1.0
        B_tor_factor = -1.0 if clockwise_B_tor else 1.0

        psi_reference = equilibrium.psi_axis # Define the magnetic axis as psi = 0

        self.CASE    = CASE
        self.NW, self.NH = psi.shape
        self.RDIM    = equilibrium.Rmax - equilibrium.Rmin
        self.ZDIM    = equilibrium.Zmax - equilibrium.Zmin
        self.RLEFT   = equilibrium.Rmin
        self.ZMID    = 0.5 * (equilibrium.Zmin + equilibrium.Zmax)
        self.RMAXIS  = equilibrium.Rgeometric()
        self.ZMAXIS  = equilibrium.Zgeometric()
        self.SIMAG   = (equilibrium.psi_axis - psi_reference) * current_factor
        self.SIBRY   = (equilibrium.psi_bndry - psi_reference) * current_factor
        self.RCENTR  = self.RMAXIS
        self.BCENTR  = (equilibrium.fvac() / self.RMAXIS) * B_tor_factor
        self.CURRENT = equilibrium.plasmaCurrent() * current_factor

        psinorm = np.linspace(0.0, 1.0, self.NW, endpoint=True)  #Includes separatrix
        self.FPOL    = equilibrium.fpol(psinorm) * B_tor_factor
        self.PRES    = equilibrium.pressure(psinorm)
        self.FFPRIM  = equilibrium.ffprime(psinorm)
        self.PPRIME  = equilibrium.pprime(psinorm)
        self.QPSI    = equilibrium.q(psinorm)
        
        self.PSIRZ   = (psi - psi_reference) * current_factor
        Rsep, Zsep = find_lcfs(equilibrium=equilibrium, n_boundary_points=n_boundary_points)

        assert Rsep.size == Zsep.size == n_boundary_points
        self.NBBBS   = n_boundary_points
        self.RBBBS   = Rsep
        self.ZBBBS   = Zsep

        wall = equilibrium.tokamak.wall
        assert wall.R.size == wall.Z.size
        self.LIMITR  = wall.R.size
        self.RLIM    = wall.R
        self.ZLIM    = wall.Z

        return self
    
    @property
    def R(self):
        """
        Returns the R coordinates of the grid [in metres]
        """
        R_min = self.RLEFT
        R_max = self.RLEFT + self.RDIM
        return np.linspace(R_min, R_max, num=self.NW)

    @property
    def Z(self):
        """
        Returns the Z coordinates of the grid [in metres]
        """
        Z_min = self.ZMID - self.ZDIM / 2.0
        Z_max = self.ZMID + self.ZDIM / 2.0
        return np.linspace(Z_min, Z_max, num=self.NH)

    @property
    def psi(self):
        """
        Returns the poloidal flux function [in Weber]
        """
        return self.PSIRZ
    
    @property
    def psi_interp(self):
        """
        Returns a bivariate interpolator for the poloidal flux function
        """
        return interpolate.RectBivariateSpline(self.R, self.Z, self.psi.T)
    
    @property
    def psi_norm(self):
        """
        Returns the normalised poloidal flux function (0 at axis, 1 at primary separatrix)
        """
        return (self.psi - self.SIMAG) / (self.SIBRY - self.SIMAG)
    
    @property
    def psi_uniform_flux_grid(self):
        """
        Returns the psi values where 1D arrays are defined
        """
        return np.linspace(self.SIMAG, self.SIBRY, num=self.NW)
    
    def B_R(self, divide_by_2pi: bool=False):
        """
        Returns the radial magnetic field [in Tesla]

        Unfortunately, there is an inconsistency in whether there should be a factor of 2*pi
        in the definition of B_R.
        * if you have a COCOS (COordinate COnventions) value and cocos > 10 then set divide_by_2pi = True
        * if not, you should always check the resulting safety factor
        """
        R_mesh, _ = np.meshgrid(self.R, self.Z)
        if not divide_by_2pi:
            return -self.psi_interp(self.R, self.Z, dy=1, grid=True).T / R_mesh
        else:
            return -self.psi_interp(self.R, self.Z, dy=1, grid=True).T / (2.0 * np.pi * R_mesh)
    
    def B_Z(self, divide_by_2pi: bool=False):
        """
        Returns the vertical magnetic field [in Tesla]

        Unfortunately, there is an inconsistency in whether there should be a factor of 2*pi
        in the definition of B_Z.
        * if you have a COCOS (COordinate COnventions) value and cocos > 10 then set divide_by_2pi = True
        * if not, you should always check the resulting safety factor
        """
        R_mesh, _ = np.meshgrid(self.R, self.Z)
        if not divide_by_2pi:
            return self.psi_interp(self.R, self.Z, dx=1, grid=True).T / R_mesh
        else:
            return self.psi_interp(self.R, self.Z, dx=1, grid=True).T / (2.0 * np.pi * R_mesh)
    
    def B_tor_vacuum(self):
        """
        Returns the vacuum toroidal magnetic field [in Tesla]
        """
        R_mesh, _ = np.meshgrid(self.R, self.Z)
        return self.BCENTR * self.RCENTR / R_mesh
    
    def B_tor_from_fpol(self):
        """
        Returns the contribution of the plasma to the toroidal field [in Tesla]

        FPOL = B_tor * R, so B_tor = FPOL / R
        """
        R_mesh, _ = np.meshgrid(self.R, self.Z)
        fpol_interp = interpolate.interp1d(self.psi_uniform_flux_grid, self.FPOL,
                                           fill_value=np.nan, bounds_error=False)
        
        return fpol_interp(self.psi) / R_mesh

    def B_tor(self):
        """
        Return the toroidal magnetic field, using the values calculated from FPOL
        in the confined region and the vacuum field outside of the confined region
        """
        B_tor_from_fpol = self.B_tor_from_fpol()
        B_tor_vacuum = self.B_tor_vacuum()
        return np.where(np.isnan(B_tor_from_fpol), B_tor_vacuum, B_tor_from_fpol)

def find_lcfs(equilibrium: freegs.equilibrium.Equilibrium, n_boundary_points: int=1001) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns arrays with R and Z points along the last-closed-flux-surface
    """
    isoflux = np.array(freegs.critical.find_separatrix(equilibrium, ntheta=n_boundary_points))
    R_lcfs, Z_lcfs, _, _ = np.split(isoflux, indices_or_sections=4, axis=1)

    # Identify which point has the lowest Z-value (usually this is the X-point)
    shift = -np.argmin(Z_lcfs)
    # Shift the array so the first element is at the lowest Z-value, and then reverse
    # so that it is ordered anticlockwise
    R_lcfs, Z_lcfs = np.roll(R_lcfs, shift)[::-1], np.roll(Z_lcfs, shift)[::-1]
    # Drop the last point, then append the first point, to make sure that the
    # _lcfseratrix closes exactly
    R_lcfs, Z_lcfs = np.append(R_lcfs[:-1], R_lcfs[0]), np.append(Z_lcfs[:-1], Z_lcfs[0])

    return R_lcfs, Z_lcfs

def read_single_values(list_of_lines, format_spec="5e16.9", start_at_line=5) -> tuple[float, int, int]:
    """
    Makes a 'generator', which is a special Python object that returns a value each time that it is called.

    This generator reads each line in the list of lines, and tries to format it according to format_spec
    For instance, the default format '5e16.9' will try to read 5 fortran-formatted floats in exponential format
    with a width of 16 characters and 9 characters after the decimal point
    Where there are less than 5 numbers on a line, the fortran reader will return 'None'. These are skipped (not
    returned by the generator)

    The generator returns the next value according to the specified format, and also the line number and position being read
    """
    for line_number, line in enumerate(list_of_lines):
        if line_number < start_at_line:
            continue

        values = FReader(f"({format_spec})").read(line)
        for value_position, value in enumerate(values):
            if value != None:
                yield value, line_number, value_position

def read_1d_array(generator, number_of_values, error_if_read_on_same_line=False) -> tuple[np.ndarray, int, int]:
    """
    Take number_of_values from the generator and write it into a 1D array
    """
    array = np.zeros(number_of_values)
    line_number, value_position = 0, 0

    for i in range(number_of_values):
        value, line_number, value_position = next(generator)

        if i == 0 and error_if_read_on_same_line:
            assert (
                value_position == 0
            ), f"Arrays are always defined as starting on a new line, but the first element in this array is not at position 0"

        array[i] = value

    return array, line_number, value_position

def read_2d_array(generator, rows, columns, fortran_order_arrays=False) -> tuple[np.ndarray, int, int]:
    """
    Take rows*columns from the generator and write it into a 2D array of
    shape (rows, columns), or (columns, rows) if transpose = True
    """
    number_of_values = rows * columns
    array, line_number, value_position = read_1d_array(generator, number_of_values)

    if fortran_order_arrays:
        array = array.reshape((rows, columns))
    else:
        array = array.reshape((columns, rows))

    return array, line_number, value_position

def write_1d_array(array_to_write: np.ndarray, element_format: str, max_line_length: int=5, array_length=None) -> str:
    """
    Sequentially write elements of an array, and return a string which can be written to file
    """
    if array_length is not None:
        assert array_to_write.size == array_length
    
    output = ""
    j = 0
    for i in range(array_to_write.size):
        output += FWriter(element_format).write([array_to_write[i]])
        j += 1
        if j == max_line_length:
            output += '\n'
            j = 0
    return output + '\n'
