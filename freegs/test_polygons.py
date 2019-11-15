from . import polygons

def test_nointersect():
    assert polygons.intersect([0,1], [0,0], [0,1], [1,1]) == False
    assert polygons.intersect([0,1], [0,0], [2,2], [1,-1]) == False

# Two line segments
def test_lineintersect():
    assert polygons.intersect([0,1], [0,0], [0.5,0.5], [1,-1]) == True

# Two squares
def test_squareintersect():
    assert polygons.intersect([0,1,1,0], [0,0,1,1],
                              [-0.5, 0.5, 0.5, -0.5], [0.5, 0.5, 1.5, 1.5]) == True
    
                            
###### clockwise

def test_clockwise():
    assert not polygons.clockwise([(0,0),(1,0),(1,1),(0,1)])
    assert  polygons.clockwise([(0,0),(0,1),(1,1),(1,0)])


