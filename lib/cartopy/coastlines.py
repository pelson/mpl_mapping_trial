import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from matplotlib.collections import PatchCollection
import matplotlib.cm
import numpy
import os

data_dir = '/data/local/sci/dev/usr/lib/python2.6/site-packages/mpl_toolkits/basemap/data/'


fname = '/data/local/dataZoo/noaaGshhs/gshhs_2.0/gshhs/gshhs_c.b'
fname = '/home/phil/Downloads/gshhs/gshhs_c.b'

def read_gshhc(filename):
    """
    Reads:
    
    Global Self-consistent Hierarchical High-resolution Shorelines
        version 2.0 July 15, 2009
        
    .. seealso:: http://www.soest.hawaii.edu/pwessel/gshhs/README.TXT
    
    TODO: Internal polygons
    
    """
    fh = open(fname, 'rb')
    
#    while True:
    for i in xrange(10000):
        header = numpy.fromfile(fh, dtype='>i4', count=11)
        if len(header) == 0:
            break
        
        points = numpy.fromfile(fh, dtype='>i4', count=header[1]*2) * 1.0e-6

        flag = header[2]
        crosses_greenwich = (flag >> 16) & 1
        points = points.reshape(-1, 2)
        lons, lats = points[:, 0], points[:, 1]
        

        if crosses_greenwich:
            # If the greenwich has been crossed, then 360 is added to any number below 0 in this format.
            # To fix this, identify any points which are more than 180 degrees apart, using this information we can identify
            # polygon groups and shift them appropriately.  
            delta = numpy.diff(lons)
            step = numpy.where(numpy.abs(delta) > 180)[0]
            step = [0] + list(step+1) + [None]
            for s1, s2 in zip(step[:-1] , step[1:]):
                if delta[s1-1] > 180:
                    lons[s1:s2] -= 360
            
        if i == 4:
            # antarctic
            lons = numpy.array(list(lons) + [lons[-1], lons[0], lons[0]])
            lats = numpy.array(list(lats) + [-90, -90, lats[0]])

        yield header, lons, lats


def coastlines_dict():
    coastline_by_id = {}
    
    for header, lons, lats in read_gshhc(fname):
        id = header[0]
        n = header[1]
        container = header[-2]
        
        flag = header[2]
        level = flag & 255
        
#        lons = numpy.append(lons, lons[0:1])
#        lats = numpy.append(lats, lats[0:1])
        
#        if id == 783:
#            print header
#            print level
#        if id < 500:
#            print header
#            print container
#            print 
        if container != -1 and (level != 3):
#            if level == 2 and n > 30: 
##                print 'level 2: ', id, container, n, numpy.vstack([lons, lats])
#                continue
#            else:
#                continue
#            print 'pre: ', coastline_by_id.get(container)
            coastline_by_id.setdefault(container, [[], []])[1].append(numpy.vstack([lons, lats]).T)
#            print '\n' * 5
#            print 'post: ', coastline_by_id[container]
#            break
        else:
#            if level == 3: print 'level 3: ', id, container
            coastline_by_id.setdefault(id, [[], []])[0].append(numpy.vstack([lons, lats]).T)
            pass
        
    return coastline_by_id 

def add_coastlines():
    """
    """
    p_verts = []
    p_codes = []
    patches = []
    for id, coastline_polygon in coastlines_dict().iteritems():
        arrs = coastline_polygon[0] + coastline_polygon[1]
        lens = numpy.array([len(arr) for arr in arrs])
        codes = numpy.ones(numpy.sum(lens)) * mpath.Path.LINETO
        codes[(numpy.cumsum(lens))[0:-1]] = mpath.Path.MOVETO
        codes[0] = mpath.Path.MOVETO
        p_verts.extend(arrs)
        p_codes.append(codes)
#        path = mpath.Path(numpy.concatenate(arrs), codes)
#        polys = mpatches.PathPatch(path)
#        if id == 5:
#            patches.append(polys)
#        patches.append(polys)
    print numpy.concatenate(p_verts)
    path = mpath.Path(numpy.concatenate(p_verts), numpy.concatenate(p_codes))
    polys = mpatches.PathPatch(path)
    patches = [polys]
    collection = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4, 
                                 facecolors='green', linewidths='0')
    plt.gca().add_collection(collection)

if __name__ == '__main__':
    import matplotlib.patches
    import cartopy.projections as prj
    f = prj.Mollweide()
    
    plt.subplot(111, projection=f)
    
    add_coastlines()
        
#    plt.grid()
    plt.show()  