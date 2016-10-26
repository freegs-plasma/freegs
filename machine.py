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


if __name__ == "__main__":
    # Run test case
    
    # Define a machine with a single coil
    coils = [{"R":2.0, "Z":"-1.5", "label":"P1", "current":3.0}]
    tokamak = Machine(coils)
    
    
