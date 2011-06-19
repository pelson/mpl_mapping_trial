import numpy as np
import matplotlib.path as mpath

def intersect_poly(path):
    """Splits a path which crosses 180 degrees into two paths, connecting each of the crossing points
    to the next - which is only really appropriate for a filled polygon."""
    a = path.vertices
    x = a[:, 0].flatten()
    
    # Split the list into groups of those on one side of 180
    groups_start = np.where(np.diff(np.sign(x-180)) != 0)[0]+1
    groups_start = [0] + list(groups_start) + [None]
    
    # XXX lots of room for optimisation here...
    odd = []
    even = []
    for i, (start, end) in enumerate(zip(groups_start[0:-1], groups_start[1:])):
        if i % 2 == 0:
            odd.append(a[start:end, :])
        else:
            even.append(a[start:end, :])
            
    # XXX Need to concatenate but follow the edge too....
    return [mpath.Path(np.concatenate(odd)),
            mpath.Path(np.concatenate(even))]


def intersect_path(path):
    """Splits a path which crosses 180 degrees into multiple paths."""
    a = path.vertices
    x = a[:, 0].flatten()
    
    # Split the list into groups of those on one side of 180
    groups_start = np.where(np.diff(np.sign(x-180)) != 0)[0]+1
    
#    print 'X', x, groups_start
#    
    if len(groups_start) == 0:
        return [path]
    
    groups_start = [0] + list(groups_start) + [None]
    
    # XXX lots of room for optimisation here...
    paths = [mpath.Path(a[start:end, :]) for i, (start, end) in enumerate(zip(groups_start[0:-1], groups_start[1:]))]
    
    # Now do some interpolation to include the intersection point
    if len(paths) > 1:
        for i, ipath in enumerate(paths[1:]):
#            print 'intersection due:', i, paths[i], ipath
            p1 = ipath.vertices
            p2 = paths[i].vertices
            import path_intersections
            intersect = path_intersections.intersecton_point(p1[-1, 0], p2[0, 0], 180, 180,
                                                                p1[-1, 1], p2[0, 1], -90, 90) 

            if intersect is None:
                raise ValueError('No intersection was identified, yet there should have been one. \nPoints: %s' % ([p1[-1, 0], p2[0, 0], 180, 180,
                                                                p1[-1, 1], p2[0, 1], -9000, 9000]) )

            intersect = np.array(intersect, ndmin=2)
            if max(p2[:, 0]) > 180:
                paths[i].vertices = np.concatenate([p2, intersect-[360, 0], ])
            else:    
                paths[i].vertices = np.concatenate([p2, intersect, ])
            
            if max(p1[:, 0]) > 180:
                ipath.vertices = np.concatenate([intersect-[360, 0], p1, ])
            else:
                ipath.vertices = np.concatenate([intersect, p1, ])
                
    return paths


if __name__ == '__main__':
    
    import matplotlib.path as mpath
    x = np.array([[0, 170, 190, 198, 182, 160], [0, 10, 20, 20, 0, 0]]).T
    p1 = mpath.Path(x)
    
    print intersect_path(p1)
#    print intersect_poly(p1)