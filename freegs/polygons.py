#
# Routines for calculating intersection or other geometric calculations
# with polygons (e.g. walls, flux surface approximations)
#

def intersect(r1, z1, r2, z2, closed1=True, closed2=True):
    """Test if two polynomials intersect. The polynomials consist of
    (r1, z1) and (r2, z2) line segments. All inputs are expected to be lists.

    Returns True or False.
    """

    assert len(r1) == len(z1)
    assert len(r2) == len(z2)

    n1 = len(r1) if closed1 else len(r1) - 1
    n2 = len(r2) if closed2 else len(r2) - 1
    
    for i in range(n1):
        for j in range(n2):
            # Test for intersection between two line segments:
            # (r1[i],z1[i]) -- (r1[i+1],z1[i+1])
            # (r1[j],z1[j]) -- (r1[j+1],z1[j+1])
            # Note that since polynomials are closed the indices wrap around
            ip = (i+1) % n1
            jp = (j+1) % n2
            
            a = r1[ip] - r1[i]
            b = r2[jp] - r2[j]
            c = z1[ip] - z1[i]
            d = z2[jp] - z2[j]

            dr = r2[jp] - r1[i]
            dz = z2[jp] - z1[i]

            det = a*d - b*c
            
            if abs(det) < 1e-6:
                continue # Almost certainly doesn't intersect
            
            alpha = (d*dr - b*dz)/det  # Location along line 1 [0,1]
            beta = (a*dz - c*dr)/det # Location along line 2 [0,1]
                
            if ((alpha > 0.0) &
                (alpha < 1.0) &
                (beta > 0.0) &
                (beta < 1.0)):
                return True
    # Tested all combinations, none intersect
    return False

def area(polygon):
    """
    Calculate the area of a polygon. Can be positive (clockwise) or negative (anticlockwise)

    Input
    
    polygon   [ (r1, z1), (r2, z2), ... ]
    """
    nvert = len(polygon) # Number of vertices
    
    # Work out the winding direction by calculating the area
    area = 0.0
    for i in range(nvert):
        r1,z1 = polygon[i]
        r2,z2 = polygon[(i+1) % nvert]
        area += (r2 - r1) * (z1 + z2) # 2*area
    return 0.5 * area

def clockwise(polygon):
    """
    Detect whether a polygon is clockwise or anti-clockwise
    True -> clockwise
    False -> anticlockwise

    Input
    
    polygon   [ (r1, z1), (r2, z2), ... ]
    """
    # Work out the winding direction by calculating the area
    return area(polygon) > 0

def triangulate(polygon):
    """
    Use the ear clipping method to turn an arbitrary polygon into triangles
    
    Input
    
    polygon   [ (r1, z1), (r2, z2), ... ]
    """

    if clockwise(polygon):
        # Copy input into list
        polygon = list(iter(polygon))
    else:
        polygon = list(reversed(polygon))
    # Now polygon should be clockwise

    nvert = len(polygon)  # Number of vertices
    assert nvert > 2
    
    triangles = []
    while nvert > 3:
        # Find an "ear"
        for i in range(nvert):
            vert = polygon[i]
            next_vert = polygon[(i+1) % nvert]
            prev_vert = polygon[i-1]

            # Take cross-product of edge from prev->vert and vert->next
            # to check whether the angle is > 180 degrees
            cross = (vert[1] - prev_vert[1])*(next_vert[0] - vert[0]) - (vert[0] - prev_vert[0])*(next_vert[1] - vert[1])
            if cross < 0:
                continue # Skip this vertex

            # Check these edges don't intersect with other edges
            r1 = [prev_vert[0], vert[0], next_vert[0]]
            z1 = [prev_vert[1], vert[1], next_vert[1]]
            
            r2 = []
            z2 = []
            if i < nvert-1:
                r2 += [ v[0] for v in polygon[(i+1):] ]
                z2 += [ v[1] for v in polygon[(i+1):] ]
            if i > 0:
                r2 += [ v[0] for v in polygon[:i] ]
                z2 += [ v[1] for v in polygon[:i] ]
            
            # (r1,z1) is the line along two edges of the triangle
            # (r2,z2) is the rest of the polygon
            if intersect(r1, z1, r2, z2, closed1=False, closed2=False):
                continue # Skip

            # Found an ear
            triangles.append([prev_vert, vert, next_vert])
            # Remove this vertex
            del polygon[i]
            nvert -= 1
            break        

    # Reduced to a single triangle
    triangles.append(polygon)
    return triangles


