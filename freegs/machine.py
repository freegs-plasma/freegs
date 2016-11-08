from .gradshafranov import Greens, GreensBr, GreensBz

from numpy import linspace

class Coil:
    """
    Represents a poloidal field coil
    """
    
    def __init__(self, R, Z, current=0.0, control=True):
        """
        R, Z - Location of the coil
        
        current - current in the coil
        control - enable or disable control system
        """
        self.R = R
        self.Z = Z
        
        self.current = current
        self.control = control

    def psi(self, R, Z):
        """
        Calculate poloidal flux at (R,Z)
        """
        return self.controlPsi(R,Z) * self.current

    def Br(self, R, Z):
        """
        Calculate radial magnetic field Br at (R,Z)
        """
        return self.controlBr(R,Z) * self.current

    def Bz(self, R, Z):
        """
        Calculate vertical magnetic field Bz at (R,Z)
        """
        return self.controlBz(R,Z) * self.current

    def controlPsi(self, R, Z):
        """
        Calculate poloidal flux at (R,Z) due to a unit current
        """
        return Greens(self.R, self.Z, R, Z)
        
    def controlBr(self, R, Z):
        """
        Calculate radial magnetic field Br at (R,Z) due to a unit current
        """
        return GreensBr(self.R,self.Z, R, Z)
        
    def controlBz(self, R, Z):
        """
        Calculate vertical magnetic field Bz at (R,Z) due to a unit current
        """
        return GreensBz(self.R,self.Z, R, Z)
        
    def __str__(self):
        return "Coil(R={0},Z={1},current={2})".format(self.R, self.Z, self.current)


class Circuit:
    """
    Represents a collection of coils connected to the same circuit
    """
    def __init__(self, coils, current=0.0, control=True):
        """
        coils - A list [ (label, Coil, multiplier) ]
        """
        self.coils = coils
        self.current = current
        self.control = control

    def psi(self, R, Z):
        """
        Calculate poloidal flux at (R,Z)
        """
        psival = 0.0
        for label, coil, multiplier in self.coils:
            coil.current = self.current * multiplier
            psival += coil.psi(R,Z)
        return psival

    def Br(self, R, Z):
        """
        Calculate radial magnetic field Br at (R,Z)
        """
        result = 0.0
        for label, coil, multiplier in self.coils:
            coil.current = self.current * multiplier
            result += coil.Br(R,Z)
        return result

    def Bz(self, R, Z):
        """
        Calculate vertical magnetic field Bz at (R,Z)
        """
        result = 0.0
        for label, coil, multiplier in self.coils:
            coil.current = self.current * multiplier
            result += coil.Bz(R,Z)
        return result

    def controlPsi(self, R, Z):
        """
        Calculate poloidal flux at (R,Z) due to a unit current
        """
        result = 0.0
        for label, coil, multiplier in self.coils:
            result += multiplier * coil.controlPsi(R, Z)
        return result
        
    def controlBr(self, R, Z):
        """
        Calculate radial magnetic field Br at (R,Z) due to a unit current
        """
        result = 0.0
        for label, coil, multiplier in self.coils:
            result += multiplier * coil.controlBr(R, Z)
        return result
        
    def controlBz(self, R, Z):
        """
        Calculate vertical magnetic field Bz at (R,Z) due to a unit current
        """
        result = 0.0
        for label, coil, multiplier in self.coils:
            result += multiplier * coil.controlBz(R, Z)
        return result
        
    def __str__(self):
        result = "Circuit( " 
        for label, coil, multiplier in self.coils:
            result += label+ ":" + str(coil)+ " "
        return result + ")"
        
    

class Solenoid:
    """
    Represents a central solenoid
    """
    def __init__(self, Rs, Zsmin, Zsmax, Ns, current=0.0, control=True):
        """
        Rs - Radius of the solenoid
        Zsmin, Zsmax - Minimum and maximum Z
        Ns - Number of turns
        
        current - current in each turn
        control - enable or disable control system
        """
        self.Rs = Rs
        self.Zsmin = Zsmin
        self.Zsmax = Zsmax
        self.Ns = Ns
        
        self.current = current
        self.control = control
        
    def psi(self, R, Z):
        """
        Calculate poloidal flux at (R,Z)
        """
        return self.controlPsi(R,Z) * self.current

    def Br(self, R, Z):
        """
        Calculate radial magnetic field Br at (R,Z)
        """
        return self.controlBr(R,Z) * self.current

    def Bz(self, R, Z):
        """
        Calculate vertical magnetic field Bz at (R,Z)
        """
        return self.controlBz(R,Z) * self.current

    def controlPsi(self, R, Z):
        """
        Calculate poloidal flux at (R,Z) due to a unit current
        
        R and Z should have the same dimensions, but can be multi-dimensional
        Return should have the same shape
        """
        result = 0.0
        for Zs in linspace(self.Zsmin, self.Zsmax, self.Ns):
            result += Greens(self.Rs, Zs, R, Z)
        return result
        
    def controlBr(self, R, Z):
        """
        Calculate radial magnetic field Br at (R,Z) due to a unit current
        """
        result = 0.0
        for Zs in linspace(self.Zsmin, self.Zsmax, self.Ns):
            result += GreensBr(self.Rs, Zs, R, Z)
        return result
        
    def controlBz(self, R, Z):
        """
        Calculate vertical magnetic field Bz at (R,Z) due to a unit current
        """
        result = 0.0
        for Zs in linspace(self.Zsmin, self.Zsmax, self.Ns):
            result += GreensBz(self.Rs, Zs, R, Z)
        return result

    def __str__(self):
        return "Solenoid(R={0},Zmin={1},Zmax={2},current={3},N={4})".format(self.Rs, self.Zsmin, self.Zsmax, self.current, self.Ns)


class Machine:
    """
    Represents the machine (Tokamak), including
    coils and power supply circuits
    
    coils[{R,Z,label,current}] - List of coils
    
    """

    def __init__(self, coils):
        """
        coils - A list of coils [(label, Coil|Circuit|Solenoid)]
        """
        self.coils = coils
    
    def psi(self, R, Z):
        """
        Poloidal flux due to coils
        """
        psi_coils = 0.0
        for label, coil in self.coils:
            psi_coils += coil.psi(R, Z)
        
        return psi_coils

    def Br(self, R, Z):
        """
        Radial magnetic field at given points
        """
        Br = 0.0;    
        for label, coil in self.coils:
            Br += coil.Br(R,Z)
        
        return Br
    
    def Bz(self, R, Z):
        """
        Vertical magnetic field
        """
        Bz = 0.0
        for label, coil in self.coils:
            Bz += coil.Bz(R, Z)
        
        return Bz

    def controlBr(self, R, Z):
        """
        Returns a list of control responses for Br
        at the given (R,Z) location(s).
        """
        return [ coil.controlBr(R,Z)
                 for label, coil in self.coils if coil.control ]

    def controlBz(self, R, Z):
        """
        Returns a list of control responses for Bz
        at the given (R,Z) location(s)
        """
        return [ coil.controlBz(R,Z)
                 for label, coil in self.coils if coil.control ]

    def controlPsi(self, R, Z):
        """
        Returns a list of control responses for psi
        at the given (R,Z) location(s)
        """
        return [ coil.controlPsi(R,Z)
                 for label, coil in self.coils if coil.control]
        

    def controlAdjust(self, current_change):
        """
        Add given currents to the controls.
        Given iterable must be the same length
        as the list returned by controlBr, controlBz
        """
        
        for lc, dI in zip(self.coils, current_change):
            label,coil = lc
            coil.current += dI
    
    def printCurrents(self):
        print("==========================")
        for label, coil in self.coils:
            print(label + " : " + str(coil))
        print("==========================")
        
        
def EmptyTokamak():
    """
    Creates a tokamak with no coils
    """
    return Machine([])

def TestTokamak():
    """
    Create a simple tokamak 
    """
    
    coils = [("P1L", Coil(1.0, -1.1)),
             ("P1U", Coil(1.0, 1.1)),
             ("P2L", Coil(1.75, -0.6)),
             ("P2U", Coil(1.75, 0.6))]
    
    return Machine(coils)


def DIIID():
    """
    PF coil set from ef20030203.d3d
    Taken from Corsica
    """

    coils = [{"label":"F1A", "R":0.8608, "Z":0.16830, "current":0.0},
             {"label":"F2A", "R":0.8614, "Z":0.50810, "current":0.0},
             {"label":"F3A", "R":0.8628, "Z":0.84910, "current":0.0},
             {"label":"F4A", "R":0.8611, "Z":1.1899 , "current":0.0},
             {"label":"F5A", "R":1.0041, "Z":1.5169 , "current":0.0},
             {"label":"F6A", "R":2.6124, "Z":0.4376 , "current":0.0},
             {"label":"F7A", "R":2.3733, "Z":1.1171 , "current":0.0},
             {"label":"F8A", "R":1.2518, "Z":1.6019 , "current":0.0},
             {"label":"F9A", "R":1.6890, "Z":1.5874 , "current":0.0},
             {"label":"F1B", "R":0.8608, "Z":-0.1737, "current":0.0},
             {"label":"F2B", "R":0.8607, "Z":-0.5135, "current":0.0},
             {"label":"F3B", "R":0.8611, "Z":-0.8543, "current":0.0},
             {"label":"F4B", "R":0.8630, "Z":-1.1957, "current":0.0},
             {"label":"F5B", "R":1.0025, "Z":-1.5169, "current":0.0},
             {"label":"F6B", "R":2.6124, "Z":-0.44376, "current":0.0},
             {"label":"F7B", "R":2.3834, "Z":-1.1171, "current":0.0},
             {"label":"F8B", "R":1.2524, "Z":-1.6027, "current":0.0},
             {"label":"F9B", "R":1.6889, "Z":-1.578, "current":0.0}]
    
    return Machine(coils)

def MAST():
    """
    Mega-Amp Spherical Tokamak. This version has all independent coils 
    so that each is powered by a separate coil
    """
    
    coils = [("P2U", Coil(0.49, 1.76)),
             ("P2L", Coil(0.49, -1.76)),
             ("P3U", Coil(1.1, 1.1)),
             ("P3L", Coil(1.1, -1.1)),
             ("P4U", Coil(1.51, 1.095)),
             ("P4L", Coil(1.51, -1.095)),
             ("P5U", Coil(1.66, 0.52)),
             ("P5L", Coil(1.66, -0.52)),
             ("P6U", Coil(1.5, 0.9)),
             ("P6L", Coil(1.5, -0.9)),
             ("P1", Solenoid(0.15, -1.24, 1.24, 100))]
    
    return Machine(coils)

def MAST_sym():
    """
    Mega-Amp Spherical Tokamak. This version the upper and lower coils 
    are connected to the same circuits P2 - P6
    """
    coils = [("P2", Circuit( [("P2U", Coil(0.49,  1.76), 1.0),
                              ("P2L", Coil(0.49, -1.76),1.0)] )),
             ("P3", Circuit( [("P3U", Coil(1.1,  1.1), 1.0),
                              ("P3L", Coil(1.1, -1.1), 1.0)] )),
             ("P4", Circuit( [("P4U", Coil(1.51,  1.095), 1.0),
                              ("P4L", Coil(1.51, -1.095), 1.0)] )),
             ("P5", Circuit( [("P5U", Coil(1.66,  0.52), 1.0),
                              ("P5L", Coil(1.66, -0.52), 1.0)] )),
             ("P6", Circuit( [("P6U", Coil(1.5,  0.9), 1.0),
                              ("P6L", Coil(1.5, -0.9), 1.0)] )),
             ("P1", Solenoid(0.15, -1.24, 1.24, 100))]
    
    return Machine(coils)

if __name__ == "__main__":
    # Run test case
    
    # Define a machine with a single coil
    coils = [{"R":2.0, "Z":"-1.5", "label":"P1", "current":3.0}]
    tokamak = Machine(coils)
    
    
