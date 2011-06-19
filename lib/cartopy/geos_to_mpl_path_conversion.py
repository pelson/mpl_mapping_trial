import cartopy.custom_projection
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.cm


from matplotlib.path import Path
from shapely.geometry.collection import GeometryCollection
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.multilinestring import MultiLineString
import shapely.geometry
from shapely.geometry import polygon, linestring, point
from shapely.geometry.multipoint import MultiPoint


def path_to_geos(path):
    """
    """
    DEBUG = False
    path_verts, path_codes =  zip(*list(path.iter_segments(curves=False)))
    path_verts = np.array(path_verts)
    path_codes = np.array(path_codes)
    
    if DEBUG: print 'codes:', path_codes
    verts_split_inds = np.where(path_codes == Path.MOVETO)[0]
    verts_split = np.split(path_verts, verts_split_inds, 0)
    codes_split = np.split(path_codes, verts_split_inds, 0)
    
    if DEBUG: print 'vs: ', `verts_split`
    if DEBUG: print 'cs: ', `codes_split`
    
    collection = []
    for path_verts, path_codes in zip(verts_split, codes_split):
        if len(path_verts) == 0:
            continue
        # XXX A path can be given which does not end with close poly, in that situation, we have to guess?
        if DEBUG: print 'pv: ', path_verts
        # XXX Implement a point
        
        # temporary fix...
#        if path_verts.shape[0] == 2:
#            continue
        
        if path_verts.shape[0] > 2 and (path_codes[-1] == Path.CLOSEPOLY or all(path_verts[0, :] == path_verts[-1, :])):
            if path_codes[-1] == Path.CLOSEPOLY:
                ipath2 = polygon.Polygon(path_verts[:-1, :])
            else:
                ipath2 = polygon.Polygon(path_verts)
        else:
            ipath2 = linestring.LineString(path_verts)
        collection.append(ipath2)
        
    i = 0
    while i<len(collection)-1:
#        for i, poly in enumerate(collection[0:-1]):
        poly = collection[i]
        poly2 = collection[i+1]
        
        # TODO Worry about islands within lakes
        if isinstance(poly, polygon.Polygon) and isinstance(poly2, polygon.Polygon):  
            if poly.contains(poly2):
                collection[i] = polygon.Polygon(poly.exterior, list(poly.interiors) + [poly2.exterior])
                collection.pop(i+1)
                continue
        i+=1
        
    if len(collection) == 1:
        return collection
    else:
        if all([isinstance(geom, linestring.LineString) for geom in collection]):
            return [MultiLineString(collection)]
        else:
            return collection
            if DEBUG: print 'geom: ', collection, type(collection)
            raise NotImplementedError('The path given was not a collection of line strings, ' 
                                      'nor a single polygon with interiors.')
        
def geos_to_path(shape):
    """
    """
    if isinstance(shape, (shapely.geometry.linestring.LineString, shapely.geometry.point.Point)):
        return [Path(np.vstack(shape.xy).T)]
    elif isinstance(shape, (shapely.geometry.multipolygon.MultiPolygon)):
        r = []
        for shp in shape:
            r.extend(geos_to_path(shp))
        return r
        
    elif isinstance(shape, (shapely.geometry.polygon.Polygon)):
        def poly_codes(poly):
            r = np.ones(len(poly.xy[0])) * Path.LINETO
            r[0] = Path.MOVETO
            return r
        
        vertices = np.concatenate([np.array(shape.exterior.xy)] + 
                                  [np.array(ring.xy) for ring in shape.interiors], 1).T
        codes = np.concatenate(
                [poly_codes(shape.exterior)]
                + [poly_codes(ring) for ring in shape.interiors])
        return [Path(vertices, codes)]
    elif isinstance(shape, (MultiPolygon, GeometryCollection, MultiLineString, MultiPoint)):
        r = []
        for geom in shape.geoms:
            r.extend(geos_to_path(geom))
        return r
#        return [Path(np.vstack(line.xy).T) for geom in shape.geoms]
    elif isinstance(shape, (GeometryCollection, MultiLineString)):
        print type(shape)
        return [Path(np.vstack(line.xy).T) for line in shape]
    else:
        raise ValueError('Unexpected GEOS type. Got %s' % type(shape))
            

if __name__ == '__main__':
                
    polys = []
    #polys.append(mpatches.RegularPolygon( (140, 10), 4, 81.0))
    #polys.append(mpatches.Polygon([[0, 45], [300, 45], [300, -45], [0, -45]]))
    pth = mpath.Path([[0, 45], [300, 45], [300, -45], [0, -45], [0, -45], [200, 20], [150, 20], [150, -20], [200, -20], [200, -20]], [1, 2, 2, 2, 79, 1, 2, 2 ,2, 79])
    #pth = mpatches.PathPatch(pth).get_path()
    
    
    g = path_to_geos(pth)
    print g
    pth = geos_to_path(g[0])[0]
    print pth
    
    polys.append(mpatches.PathPatch(pth))
    
    collection = PatchCollection(polys, cmap=matplotlib.cm.jet, alpha=0.4)
    
    plt.gca().add_collection(collection)
    plt.gca().set_xlim(-250, 250)
    plt.gca().set_ylim(-250, 250)
    plt.show()