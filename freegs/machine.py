"""
Classes and routines to represent coils and circuits

License
-------

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

from .gradshafranov import Greens, GreensBr, GreensBz

from numpy import linspace, dtype

class Coil:
    """
    Represents a poloidal field coil

    public members
    --------------
    
    R, Z - Location of the coil
    current - current in the coil
    turns   - Number of turns
    control - enable or disable control system

    The total toroidal current carried by the coil is current * turns
    """
    
    def __init__(self, R, Z, current=0.0, turns=1, control=True):
        """
        R, Z - Location of the coil
        
        current - current in the coil
        control - enable or disable control system
        """
        self.R = R
        self.Z = Z
        
        self.current = current
        self.turns = turns
        self.control = control

    def psi(self, R, Z):
        """
        Calculate poloidal flux at (R,Z)
        """
        return self.controlPsi(R,Z) * self.current

    def createPsiGreens(self, R, Z):
        """
        Calculate the Greens function at every point, and return
        array. This will be passed back to evaluate Psi in
        calcPsiFromGreens()
        """
        return self.controlPsi(R,Z)

    def calcPsiFromGreens(self, pgreen):
        """
        Calculate plasma psi from Greens functions and current
        """
        return self.current * pgreen

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
        return Greens(self.R, self.Z, R, Z) * self.turns
        
    def controlBr(self, R, Z):
        """
        Calculate radial magnetic field Br at (R,Z) due to a unit current
        """
        return GreensBr(self.R,self.Z, R, Z) * self.turns
        
    def controlBz(self, R, Z):
        """
        Calculate vertical magnetic field Bz at (R,Z) due to a unit current
        """
        return GreensBz(self.R,self.Z, R, Z) * self.turns
        
    def __repr__(self):
        return ("Coil(R={0}, Z={1}, current={2}, turns={3}, control={4})"
                .format(self.R, self.Z, self.current, self.turns, self.control))

    def __eq__(self, other):
        return (self.R == other.R
                and self.Z == other.Z
                and self.current == other.current
                and self.turns == other.turns
                and self.control == other.control)

    def __ne__(self, other):
        return not self == other

    def to_tuple(self):
        """
        Helper method for writing output
        """
        return (self.R, self.Z, self.current, self.turns, self.control)


class Circuit:
    """
    Represents a collection of coils connected to the same circuit
    
    Public members
    --------------
    
    current  Current in the circuit [Amps]
    control  Use feedback control? [bool]
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

    def createPsiGreens(self, R, Z):
        """
        Calculate Greens functions
        """
        pgreen = {}
        for label, coil, multiplier in self.coils:
            pgreen[label] = coil.createPsiGreens(R, Z)
        return pgreen

    def calcPsiFromGreens(self, pgreen):
        """
        Calculate psi from Greens functions
        
        """
        psival = 0.0
        for label, coil, multiplier in self.coils:
            coil.current = self.current * multiplier
            psival += coil.calcPsiFromGreens(pgreen[label])
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

    def __repr__(self):
        result = "Circuit(["
        coils = ['("{0}", {1}, {2})'.format(label, coil, multiplier)
                 for label, coil, multiplier in self.coils]
        result += ", ".join(coils)
        return result + "], current={0}, control={1})".format(self.current, self.control)

    def __eq__(self, other):
        return (self.coils == other.coils
                and self.current == other.current
                and self.control == other.control)

    def __ne__(self, other):
        return not self == other

    def to_tuple(self):
        """
        Helper method for writing output
        """
        return [(label, coil.to_tuple(), multiplier)
                 for label, coil, multiplier in self.coils]


class Solenoid:
    """
    Represents a central solenoid

    Public members
    --------------
    
    current - current in each turn
    control - enable or disable control system
    
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

    def createPsiGreens(self, R, Z):
        """
        Calculate Greens functions
        """
        return self.controlPsi(R,Z)

    def calcPsiFromGreens(self, pgreen):
        """
        Calculate psi from Greens functions
        """
        return self.current * pgreen

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

    def __repr__(self):
        return ("Solenoid(Rs={0}, Zsmin={1}, Zsmax={2}, current={3}, Ns={4}, control={5})"
                .format(self.Rs, self.Zsmin, self.Zsmax, self.current, self.Ns, self.control))

    def __eq__(self, other):
        return (self.Rs == other.Rs
                and self.Zsmin == other.Zsmin
                and self.Zsmax == other.Zsmax
                and self.Ns == other.Ns
                and self.current == other.current
                and self.control == other.control)

    def __ne__(self, other):
        return not self == other

    def to_tuple(self):
        """
        Helper method for writing output
        """
        return (self.Rs, self.Zsmin, self.Zsmax, self.Ns, self.current, self.control)


class Wall:
    """
    Represents the wall of the device. 
    Consists of an ordered list of (R,Z) points
    """
    def __init__(self, R, Z):
        self.R = R
        self.Z = Z

    def __repr__(self):
        return "Wall(R={R}, Z={Z})".format(R=self.R, Z=self.Z)

    def __eq__(self, other):
        return (self.R == other.R
                and self.Z == other.Z)

    def __ne__(self, other):
        return not self == other
    
    
class Machine:
    """
    Represents the machine (Tokamak), including
    coils and power supply circuits
    
    coils[(label, Coil|Circuit|Solenoid] - List of coils
    
    Note: a list is used rather than a dict, so that the coils
    remain ordered, and so can be updated easily by the control system.
    Instead __getitem__ is implemented to allow access to coils

    """

    def __init__(self, coils, wall=None):
        """
        coils - A list of coils [(label, Coil|Circuit|Solenoid)]
        """
        
        self.coils = coils
        self.wall = wall

    def __repr__(self):
        return ("Machine(coils={coils}, wall={wall})"
                .format(coils=self.coils, wall=self.wall))

    def __eq__(self, other):
        # Other Machine might be equivalent except for order of
        # coils. Assume this doesn't actually matter
        return (sorted(self.coils) == sorted(other.coils)
                and self.wall == other.wall)

    def __ne__(self, other):
        return not self == other

    def __getitem__(self, name):
        for label, coil in self.coils:
            if label == name:
                return coil
        raise KeyError("Machine does not contain coil with label '{0}'".format(name))
        
    def psi(self, R, Z):
        """
        Poloidal flux due to coils
        """
        psi_coils = 0.0
        for label, coil in self.coils:
            psi_coils += coil.psi(R, Z)
        
        return psi_coils

    def createPsiGreens(self, R, Z):
        """
        An optimisation, which pre-computes the Greens functions
        and puts into arrays for each coil. This map can then be
        called at a later time, and quickly return the field
        """
        pgreen = {}
        for label, coil in self.coils:
            pgreen[label] = coil.createPsiGreens(R, Z)
        return pgreen
        
    def calcPsiFromGreens(self, pgreen):
        """
        Uses the object returned by createPsiGreens to quickly
        compute the plasma psi
        """
        psi_coils = 0.0
        for label, coil in self.coils:
            psi_coils += coil.calcPsiFromGreens(pgreen[label])
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
        # Get list of coils being controlled
        controlcoils = [coil for label,coil in self.coils if coil.control]
        
        for coil, dI in zip(controlcoils, current_change):
            coil.current += dI

    def controlCurrents(self):
        """
        Return a list of coil currents for the coils being controlled
        """
        return [ coil.current for label, coil in self.coils if coil.control ]

    def setControlCurrents(self, currents):
        """
        Sets the currents in the coils being controlled.
        Input list must be of the same length as the list
        returned by controlCurrents
        """
        controlcoils = [coil for label,coil in self.coils if coil.control]
        for coil, current in zip(controlcoils, currents):
            coil.current = current
            
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

    wall = Wall([ 0.75, 0.75,  1.5,  1.8,   1.8,   1.5],   # R
                [-0.85, 0.85, 0.85, 0.25, -0.25, -0.85])   # Z
    
    return Machine(coils, wall)


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
             ("P6L", Coil(1.5, -0.9))
             ,("P1", Solenoid(0.15, -1.4, 1.4, 100))
             ] 
    
    return Machine(coils)

def MAST_sym():
    """
    Mega-Amp Spherical Tokamak. This version the upper and lower coils 
    are connected to the same circuits P2 - P6
    """
    coils = [("P2", Circuit( [("P2U", Coil(0.49,  1.76), 1.0),
                              ("P2L", Coil(0.49, -1.76),1.0)] ))
             ,("P3", Circuit( [("P3U", Coil(1.1,  1.1), 1.0),
                              ("P3L", Coil(1.1, -1.1), 1.0)] ))
             ,("P4", Circuit( [("P4U", Coil(1.51,  1.095), 1.0),
                              ("P4L", Coil(1.51, -1.095), 1.0)] ))
             ,("P5", Circuit( [("P5U", Coil(1.66,  0.52), 1.0),
                              ("P5L", Coil(1.66, -0.52), 1.0)] ))
             ,("P6", Circuit( [("P6U", Coil(1.5,  0.9), 1.0),
                               ("P6L", Coil(1.5, -0.9), -1.0)] ))
             ,("P1", Solenoid(0.15, -1.45, 1.45, 100))
         ]
    
    return Machine(coils)

def TCV():
    coils = [
        #             ("A1", Coil(0.422500,0.000000)),
        #             ("B1", Coil(0.445700,-0.936000)),
        #             ("B2", Coil(0.445700,0.936000)),
        #             ("C1", Coil(0.621500,-1.110000)),
        #             ("C2", Coil(0.621500,1.110000)),
        #             ("D1", Coil(1.176500,-1.170000)),
        #             ("D2", Coil(1.176500,1.170000)),
        ("E1", Coil(0.505000,-0.700000)),
        ("E2", Coil(0.505000,-0.500000)),
        ("E3", Coil(0.505000,-0.300000)),
        ("E4", Coil(0.505000,-0.100000)),
        ("E5", Coil(0.505000,0.100000)),
        ("E6", Coil(0.505000,0.300000)),
        ("E7", Coil(0.505000,0.500000)),
        ("E8", Coil(0.505000,0.700000)),
        ("F1", Coil(1.309500,-0.770000)),
        ("F2", Coil(1.309500,-0.610000)),
        ("F3", Coil(1.309500,-0.310000)),
        ("F4", Coil(1.309500,-0.150000)),
        ("F5", Coil(1.309500,0.150000)),
        ("F6", Coil(1.309500,0.310000)),
        ("F7", Coil(1.309500,0.610000)),
        ("F8", Coil(1.309500,0.770000)),
        #             ("G1", Coil(1.098573,-0.651385)),
        #             ("G2", Coil(1.114000,-0.633000)),
        #             ("G3", Coil(1.129427,-0.614615)),
        #             ("G4", Coil(1.129427,0.614615)),
        #             ("G5", Coil(1.114000,0.633000)),
        #             ("G6", Coil(1.098573,0.651385)),
        ("T1", Coil(1.554000,-0.780000)),
        ("T2", Coil(1.717000,-0.780000)),
        ("T3", Coil(1.754000,-0.780000)),
        ("OH", Circuit([("OH1",Solenoid(0.43,-0.93,0.93,100),0.01),
                        ("C1", Coil(0.621500,-1.110000), 1.0),
                        ("C2", Coil(0.621500,1.110000), 1.0),
                        ("D1", Coil(1.176500,-1.170000), 1.0),
                        ("D2", Coil(1.176500,1.170000), 1.0)  ]))
    ]
    
    return Machine(coils)

def MASTU():
    coils = [
        ("Solenoid", Solenoid(0.19475, -1.581, 1.581, 324)),
        ("Pc", Coil(0.067, 0.0, turns=142)),
        ("Px", Circuit([("PxU", Coil(0.2405,  1.2285, turns=44), 1.0),
                        ("PxL", Coil(0.2405, -1.2285, turns=44), 1.0)])),
        ("D1", Circuit([("D1U", Coil(0.381,  1.555, turns=35), 1.0),
                        ("D1L", Coil(0.381, -1.555, turns=35), 1.0)])),
        ("D2", Circuit([("D2U", Coil(0.574,  1.734, turns=23), 1.0),
                        ("D2L", Coil(0.574, -1.734, turns=23), 1.0)])),
        ("D3", Circuit([("D3U", Coil(0.815,  1.980, turns=23), 1.0),
                        ("D3L", Coil(0.815, -1.980, turns=23), 1.0)])),
        ("Dp", Circuit([("DpU", Coil(0.918,  1.501, turns=23), 1.0),
                        ("DpL", Coil(0.918, -1.501, turns=23), 1.0)])),
        ("D5", Circuit([("D5U", Coil(1.900,  1.950, turns=27), 1.0),
                        ("D5L", Coil(1.900, -1.950, turns=27), 1.0)])),
        ("D6", Circuit([("D6U", Coil(1.285,  1.470, turns=23), 1.0),
                        ("D6L", Coil(1.285, -1.470, turns=23), 1.0)])),
        ("D7", Circuit([("D7U", Coil(1.520,  1.470, turns=23), 1.0),
                        ("D7L", Coil(1.520, -1.470, turns=23), 1.0)])),
        ("P4", Circuit([("P4U", Coil(1.500,  1.100, turns=23), 1.0),
                        ("P4L", Coil(1.500, -1.100, turns=23), 1.0)])),
        ("P5", Circuit([("P5U", Coil(1.650,  0.357, turns=23), 1.0),
                        ("P5L", Coil(1.650, -0.357, turns=23), 1.0)])),

        # Vertical control coils wired in opposite directions
        # Two pairs of coils, P6 pair 1 and P6 pair 2
        ("P61", Circuit([("P61U", Coil(1.1975,  1.11175, turns=2),  1.0),
                         ("P61L", Coil(1.1975, -1.11175, turns=2), -1.0)])),
        ("P62", Circuit([("P62U", Coil(1.2575,  1.0575, turns=2),  1.0),
                         ("P62L", Coil(1.2575, -1.0575, turns=2), -1.0)]))
        ]

    rwall = [1.2503, 1.3483, 1.47, 1.47, 1.45, 1.45, 1.3214, 1.1904,
             0.89296, 0.86938, 0.83981, 0.82229, 0.81974, 0.81974,
             0.82734, 0.8548, 0.89017, 0.91974, 0.94066, 1.555, 1.85,
             2, 2, 2, 2, 1.3188, 1.7689, 1.7301, 1.35, 1.09, 1.09,
             0.90576, 0.87889, 0.87889, 0.90717, 0.53948, 0.5112,
             0.50553, 0.53594, 0.5074, 0.4974, 0.5074, 0.4788, 0.4688,
             0.4788, 0.333, 0.333, 0.275, 0.334, 0.261, 0.261, 0.244,
             0.261, 0.261, 0.244, 0.261, 0.261]

    zwall = [1, 0.86, 0.86, 0.81, 0.81, 0.82, 0.82, 1.007, 1.304,
             1.3312, 1.3826, 1.4451, 1.4812, 1.4936, 1.5318, 1.5696,
             1.5891, 1.5936, 1.5936, 1.567, 1.08, 1.08, 1.7, 2.035,
             2.169, 2.169, 1.7189, 1.68, 2.06, 2.06, 2.06, 1.8786,
             1.9055, 1.9055, 1.8772, 1.5095, 1.5378, 1.5321, 1.5017,
             1.4738, 1.4838, 1.4738, 1.4458, 1.4558, 1.4458, 1.303,
             1.1, 1.1, 1.1, 0.502, 0.348, 0.348, 0.348, 0.146, 0.146, 0.146, 0]

    # Concatenate with mirror image in Z,
    # with points in reverse order
    rwall = rwall + rwall[::-1]
    zwall = zwall + [-z for z in zwall[::-1]]
    
    return Machine(coils, Wall(rwall, zwall))


if __name__ == "__main__":
    # Run test case
    
    # Define a machine with a single coil
    coils = [{"R":2.0, "Z":"-1.5", "label":"P1", "current":3.0}]
    tokamak = Machine(coils)
    
    
