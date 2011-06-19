import numpy as np

def intersection_points(p1, p2):
    intersections = []
    
    p1_segments = list(p1.iter_segments(curves=False))
    p2_segments = list(p2.iter_segments(curves=False))
    for i in xrange(len(p1_segments)-1):
        # XXX Worry about segment type
        p1_0 = p1_segments[i][0]
        p1_1 = p1_segments[i+1][0]
    
        for j in xrange(len(p2_segments)-1):
            # XXX Worry about segment type
            p2_0 = p2_segments[j][0]
            p2_1 = p2_segments[j+1][0]
         
            print p1_0, p1_1, p2_0, p2_1

            print 'inter: ', intersecton_point(p1_0[0], p1_1[0], p2_0[0], p2_1[0], p1_0[1], p1_1[1], p2_0[1], p2_1[1])
            
    return intersections

def intersecton_point(x1, x2, x3, x4, y1, y2, y3, y4, strict=True):
    # Returns the intersection point of the two lines (see http://mathworld.wolfram.com/Line-LineIntersection.html)
    # if not strict then the intersection point may not be contained on either of the given lines, if strict
    # then the point will be contained in BOTH (or return None). 
    line1_det = x1 * y2 - y1 * x2
    line2_det = x3 * y4 - y3 * x4
    
    divisor = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if divisor == 0:
        # Parallel lines...???
        return None
    
    x = (
         (line1_det * (x3 - x4) - line2_det * (x1 - x2)) / 
         divisor
         )
    y = (
         (line1_det * (y3 - y4) - line2_det * (y1 - y2)) /
         divisor
         )
    
    if strict:
        on_line_1 = min([x1, x2]) <= x <= max([x1, x2]) and \
                    min([y1, y2]) <= y <= max([y1, y2])
        on_line_2 = min([x3, x4]) <= x <= max([x3, x4]) and \
                    min([y3, y4]) <= y <= max([y3, y4])
        
        if on_line_1 and on_line_2:
            return x, y
        else:
            return None
        
    return x, y


#if __name__ == '__main__':
#    
#    import matplotlib.path as mpath
#    x = np.array([[0, 10, 20, 5], [0, 10, 20, 5]]).T
#    p1 = mpath.Path(x)
#    
#    y = np.array([[10, 0], [0, 10]]).T
#    p2 = mpath.Path(y)
#    
##    print intersection_points(p1, p2)
#
#
#    import shapely.geometry as geom
#    geom1 = geom.Polygon(p1.vertices)
#    geom2 = geom.LineString(p2.vertices)
#
#    print geom2.union(geom1)
#    print geom1.in(geom2)