#
# Routines for calculating intersection or other geometric calculations
# with polygons (e.g. walls, flux surface approximations)
#

def intersect(r1, z1, r2, z2):
    """Test if two closed polynomials intersect. The polynomials consist of
    (r1, z1) and (r2, z2) line segments. All inputs are expected to be lists.

    Returns True or False.
    """

    assert len(r1) == len(z1)
    assert len(r2) == len(z2)

    n1 = len(r1)
    n2 = len(r2)
    
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

            
    
