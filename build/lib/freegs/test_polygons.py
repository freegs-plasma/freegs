from . import polygons
import numpy as np


def test_nointersect():
    assert not polygons.intersect([0, 1], [0, 0], [0, 1], [1, 1])
    assert not polygons.intersect([0, 1], [0, 0], [2, 2], [1, -1])


# Two line segments
def test_lineintersect():
    assert polygons.intersect([0, 1], [0, 0], [0.5, 0.5], [1, -1])


# Two squares
def test_squareintersect():
    assert polygons.intersect(
        [0, 1, 1, 0], [0, 0, 1, 1], [-0.5, 0.5, 0.5, -0.5], [0.5, 0.5, 1.5, 1.5]
    )


def test_area():
    assert np.isclose(polygons.area([(0, 0), (0, 1), (1, 1), (1, 0)]), 1.0)
    assert np.isclose(polygons.area([(0, 0), (0, 1), (1, 0)]), 0.5)


# clockwise


def test_clockwise():
    assert not polygons.clockwise([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert polygons.clockwise([(0, 0), (0, 1), (1, 1), (1, 0)])


# triangulate


def test_triangulate_three():
    # Always returns a triangle in clockwise order
    assert polygons.triangulate([(0, 0), (0, 1), (1, 0)]) == [[(0, 0), (0, 1), (1, 0)]]
    assert polygons.clockwise(polygons.triangulate([(0, 0), (1, 0), (0, 1)])[0])


def test_triangulate_four():
    square = [(0, 0), (0, 1), (1, 1), (1, 0)]
    result = polygons.triangulate(square)
    assert len(result) == 2  # Two triangles
    assert len(square) == 4  # Original not modified
    # Both triangles have positive area
    assert polygons.area(result[0]) > 0.0
    assert polygons.area(result[1]) > 0.0
    # Sum of areas equal to original area
    assert np.isclose(
        polygons.area(square), polygons.area(result[0]) + polygons.area(result[1])
    )
