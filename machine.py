from gradshafranov import Greens, GreensBr, GreensBz

class Machine:
    """
    Represents the machine (Tokamak), including
    coils and power supply circuits
    
    coils[{R,Z,label,current}] - List of coils
    
    """

    def __init__(self, coils):
        self.coils = coils
    
    def psi(self, R, Z):
        """
        Poloidal flux due to coils
        """
        psi_coils = 0.0
        for coil in self.coils:
            psi_coils += coil["current"] * Greens(coil["R"], coil["Z"], R, Z)
        return psi_coils

    def Br(self, R, Z):
        """
        Radial magnetic field at given points
        """
        Br = 0.0;    
        for coil in self.coils:
            Br += coil["current"]*GreensBr(coil["R"],
                                           coil["Z"],
                                           R, Z)
        return Br
    
    def Bz(self, R, Z):
        """
        Vertical magnetic field
        """
        Bz = 0.0
        for coil in self.coils:
            Bz += coil["current"]*GreensBz(coil["R"],
                                           coil["Z"],
                                           R, Z)
        return Bz

    def controlBr(self, R, Z):
        """
        Returns a list of control responses for Br
        at the given (R,Z) location(s).
        """
        return [ GreensBr(coil["R"], coil["Z"],
                          R, Z)
                 for coil in self.coils ]

    def controlBz(self, R, Z):
        """
        Returns a list of control responses for Bz
        at the given (R,Z) location(s)
        """
        return [ GreensBz(coil["R"], coil["Z"],
                          R, Z)
                 for coil in self.coils ]

    def controlPsi(self, R, Z):
        """
        Returns a list of control responses for psi
        at the given (R,Z) location(s)
        """
        return [ Greens(coil["R"], coil["Z"],
                        R, Z)
                 for coil in self.coils ]
        

    def controlAdjust(self, current_change):
        """
        Add given currents to the controls.
        Given iterable must be the same length
        as the list returned by controlBr, controlBz
        """
        if len(current_change) != len(self.coils):
            raise ValueError("Number of current adjustments incorrect: expected %d, got %d" % (len(self.coils), len(currents)))
        
        for coil, dI in zip(self.coils, current_change):
            coil["current"] += dI
    
    def printCurrents(self):
        print("==========================")
        for c in self.coils:
            print("%s : (%.2f,%.2f)  %e" % ( c["label"], c["R"], c["Z"], c["current"]))
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
    
    coils = [{"R":1.0, "Z":-1.1, "label":"P1L", "current":0.0},
             {"R":1.0, "Z":1.1, "label":"P1U", "current":0.0},
             {"R":1.75, "Z": -0.6, "label":"P2L", "current":0.0},
             {"R":1.75, "Z": 0.6, "label":"P2U", "current":0.0}]
    
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
    Mega-Amp Spherical Tokamak
    """
    coils = [{"label":"P2U", "R":0.5, "Z":1.6, "current":0.0},
             {"label":"P2L", "R":0.5, "Z":-1.6, "current":0.0},
             {"label":"P3U", "R":1.1, "Z":1.1, "current":0.0},
             {"label":"P3L", "R":1.1, "Z":-1.1, "current":0.0},
             {"label":"P4U", "R":1.5, "Z":1.1, "current":0.0},
             {"label":"P4L", "R":1.5, "Z":-1.1, "current":0.0},
             {"label":"P5U", "R":1.6, "Z":0.5, "current":0.0},
             {"label":"P5L", "R":1.6, "Z":-0.5, "current":0.0},
             {"label":"P6U", "R":1.5, "Z":0.9, "current":0.0},
             {"label":"P6L", "R":1.5, "Z":-0.9, "current":0.0}]
    return Machine(coils)

if __name__ == "__main__":
    # Run test case
    
    # Define a machine with a single coil
    coils = [{"R":2.0, "Z":"-1.5", "label":"P1", "current":3.0}]
    tokamak = Machine(coils)
    
    
