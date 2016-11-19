"""
Unit tests of G-EQDSK input and output
"""

import unittest
import numpy

try:
    from cStringIO import StringIO
except:
    from StringIO import StringIO

import _geqdsk

class GeqdskTest(unittest.TestCase):

    def test(self):
        
        nx = 65
        ny = 65
        
        # Create a dummy dataset
        data = {"nx":nx, "ny":ny,
                "rdim":2.0,
                "zdim":1.5,
                "rcentr":1.2,
                "bcentr":2.42,
                "rleft": 0.5,
                "zmid":0.1,
                "rmagx": 1.1,
                "zmagx": 0.2,
                "simagx": -2.3,
                "sibdry": 0.21,
                "cpasma": 1234521,
                "fpol":numpy.random.rand(nx),
                "pres":numpy.random.rand(nx),
                "qpsi":numpy.random.rand(nx),
                "psi":numpy.random.rand(nx,ny)
                }

        output = StringIO()

        # Write to string
        _geqdsk.write(data, output)

        # Move to the beginning of the buffer
        output.seek(0)
        
        # Read from string
        data2 = _geqdsk.read(output)

        # Check that data and data2 are the same
        for key in data:
            numpy.testing.assert_allclose(data2[key], data[key])
        

if __name__ == '__main__':
    unittest.main()
    
