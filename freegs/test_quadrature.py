import numpy as np

from . import quadrature

def test_unity():
    # Sum of weights is one i.e quadrature works out average over area

    for n in [1,3,6]:
        ps = quadrature.polygon_quad([(0,0), (0,1), (1,1), (1,0)], n=n)
        assert len(ps) == 2 * n # Two triangles
        assert np.isclose(sum(weight for r,z,weight in ps), 1.0)

