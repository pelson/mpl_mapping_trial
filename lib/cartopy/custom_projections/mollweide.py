import numpy as np

import matplotlib
from mpl_toolkits.mplot3d.art3d import mpath
from shapely.geometry.collection import GeometryCollection
from shapely.geometry.multilinestring import MultiLineString
import shapely.geometry

import cartopy.geos_to_mpl_path_conversion as shape_convert

from shapely.geometry import polygon, linestring, point
from shapely.geometry.multipolygon import MultiPolygon
rcParams = matplotlib.rcParams
from matplotlib.axes import Axes
from matplotlib import cbook
from matplotlib.patches import Circle
from matplotlib.path import Path
import matplotlib.spines as mspines
import matplotlib.axis as maxis
from matplotlib.ticker import Formatter, Locator, NullLocator, FixedLocator, NullFormatter, LinearLocator
from matplotlib.transforms import Affine2D, Affine2DBase, Bbox, \
    BboxTransformTo, IdentityTransform, Transform, TransformWrapper
    
    
class MollweideTransform(Transform):
    """
    The base Mollweide transform.
    """
    input_dims = 2
    output_dims = 2
    is_separable = False

    def __repr__(self):
        return 'Mollweide....'
    
    def __str__(self):
        return repr(self)
    
    def __init__(self, projection, resolution):
        """
        Create a new Mollweide transform.  Resolution is the number of steps
        to interpolate between each input line segment to approximate its
        path in curved Mollweide space.
        """
        self.projection = projection
        Transform.__init__(self)
        self._resolution = resolution

    def transform_lon_lat(self, longitude, latitude):
        x, y = self.projection.forward(longitude, latitude, radians=False)
        return x, y

    def transform(self, ll):
        longitude = ll[:, 0:1]
        latitude  = ll[:, 1:2]
        
        longitude = longitude.copy()
        # Scale the range
#        longitude += 180
#        longitude %= 360
#        longitude -= 180
        
        x, y = self.transform_lon_lat(longitude, latitude)
        r = np.concatenate([x, y], 1)

#        print 'll: ', ll
#        print 'xy: ', r
#        print '------------'
        return r

    def cut_path(self, path):
        import custom_projections.numpy_intersect
        paths = custom_projections.numpy_intersect.intersect_path(path)        
        return paths
        

    def transform_no_mod(self, ll):
        longitude = ll[:, 0:1]
        latitude  = ll[:, 1:2]
        return np.concatenate(self.transform_lon_lat(longitude, latitude), 1)

    def transform_point(self, point):
#        print 'transform point: ', point
        x, y = self.transform_lon_lat(point[0], point[1])
        if x > 180:
            x -= 360
        return x, y

    def transform_path(self, path):
        
        
        def path_interpolation(path, n_steps):
            path_verts, path_codes =  zip(*list(path.iter_segments(curves=False)))
            path_verts = np.array(path_verts)
            path_codes = np.array(path_codes)
            verts_split_inds = np.where(path_codes == Path.MOVETO)[0]
            verts_split = np.split(path_verts, verts_split_inds, 0)
            codes_split = np.split(path_codes, verts_split_inds, 0)
            
            v_collection = []
            c_collection = []
            for path_verts, path_codes in zip(verts_split, codes_split):
                if len(path_verts) == 0:
                    continue
                import matplotlib.cbook
#                print path_verts.shape
                verts = matplotlib.cbook.simple_linear_interpolation(path_verts, n_steps)
                v_collection.append(verts)
#                print verts.shape
                codes = np.ones(verts.shape[0]) * Path.LINETO
                codes[0] = Path.MOVETO
                c_collection.append(codes)
                
            return Path(np.concatenate(v_collection), np.concatenate(c_collection))
            
#        print 'transform path:', path
#        print 'p shape: ', path.vertices, path.vertices.shape
        if path.vertices.shape == (1, 2):
            return Path(self.transform_no_mod(path.vertices))
        
        
#        print '\n'.join(['%s %s' % (point, code) for point, code in path.iter_segments()])
        
        p2 = polygon.Polygon(((-180, -90), (180, -90), (180, 90), (-180, 90)))
        
        paths = []
        shps = []
        for shp in shape_convert.path_to_geos(path):
            shps = []
            for lon in range(-360, 720, 360):
#            for lon in range(0, 360, 360):
                if isinstance(shp, polygon.Polygon):
                    c_shp = polygon.Polygon(np.array(shp.exterior) - [lon, 0], 
                                            [np.array(ring) - [lon, 0] for ring in shp.interiors])
                elif isinstance(shp, linestring.LineString):
                    c_shp = linestring.LineString(np.array(shp) - [lon, 0])
                elif isinstance(shp, MultiLineString):
                    c_shp = MultiLineString([linestring.LineString(np.array(s) - [lon, 0]) for s in shp.geoms])
                else:
                    raise ValueError('Unknown shapely object (%s).' % (type(shp)))
                
                shps.append(c_shp)
                
            if isinstance(shp, polygon.Polygon):
                shps = MultiPolygon(shps)
            elif isinstance(shp, linestring.LineString):
                shps = MultiLineString(shps)
            else:
                ValueError('Unknown shape type')
            
            # join the shapes back together again if they have wrapped the entire 360 range.
            from shapely.ops import cascaded_union
            shps = cascaded_union(shps)        
            interp_resolution = 40
    
#            try:
            intersection_shp = shps.intersection(p2)
            pths = shape_convert.geos_to_path(intersection_shp)
#            print pths
            
            paths.extend([path_interpolation(pth, interp_resolution) for pth in pths])
#            print paths
#            except shapely.geos.TopologicalError:
#                print 'failed with: ', shps
#                print 'orig path: ', path
#                print 'path: ', shape_convert.geos_to_path(shps)
#                import matplotlib.pyplot as plt
#                from matplotlib.collections import PatchCollection
#                import matplotlib.patches as mpatches
#                import matplotlib.cm
#
#                plt.close()
#                pth = shape_convert.geos_to_path(shps)[0]
#                poly = mpatches.PathPatch(pth)
#                collection = PatchCollection([poly], cmap=matplotlib.cm.jet, alpha=0.4)
#                plt.gca().add_collection(collection)
#                plt.show()
    
        if len(paths) == 1:
            path = paths[0]
        elif len(paths) == 0:
            return Path(np.empty([0,2]))
        else:
            points = []
            codes = []
            for path in paths:
                path_points, path_codes = zip(*(path.iter_segments()))
                points.append(path_points)
                codes.append(path_codes)

            points = [np.array(pts) for pts in points]
            path = Path(np.concatenate(points, 0), np.concatenate(codes))
#        print '------------'
#        print '\n'.join(['%s %s' % (point, code) for point, code in path.iter_segments()])
#        print 'll post: ', path.vertices
#        print path.codes
#        path = path.interpolated(40)
#        path = path.interpolated(4)
#        print path.vertices
#        print ']]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]'
        path.vertices = self.transform_no_mod(path.vertices)

        return path
    
    transform_path.__doc__ = Transform.transform_path.__doc__

    def transform_path_non_affine(self, path):
        print 'non affine path'
    
    def __array__(self, *args, **kwargs):
        # dont know what this does...
        print args, kwargs
        raise NotImplementedError()
    
    def inverted(self):
        return InvertedMollweideTransform(self.projection, self._resolution)
    inverted.__doc__ = Transform.inverted.__doc__


class InvertedMollweideTransform(Transform):
    input_dims = 2
    output_dims = 2
    is_separable = False

    def __init__(self, projection, resolution):
        self.projection = projection
        Transform.__init__(self)
        self._resolution = resolution

    def transform_x_y(self, x, y):
        return self.projection.inverse(x, y, radians=False)
    
    def transform(self, xy):
        lon, lat = self.transform_x_y(xy[:, 0], xy[:, 1])
        r = np.concatenate([lon, lat], 1)
        r.shape = xy.shape 
#        print 'ndim: ', r.ndim
#        print xy.ndim
#        print xy.shape
        
        return r    
    transform.__doc__ = Transform.transform.__doc__

    def inverted(self):
        return MollweideAxes.MollweideTransform(self._resolution)
    inverted.__doc__ = Transform.inverted.__doc__


import cartopy.custom_projection
class MollweideAxes(cartopy.custom_projection.GeoAxes):
    
    name = 'mollweide2'
    def __init__(self, *args, **kwargs):
        import cartopy.projections
        self.projection = cartopy.projections.Mollweide()
        self._latitude_cap = np.pi / 2.0
        GeoAxes.__init__(self, *args, **kwargs)
        self.set_aspect(0.5, adjustable='box', anchor='C')
        self.cla()

    def __str__(self):
        return 'Mollwide AXES!!!!'
    
    def _get_core_transform(self, resolution):
        return MollweideTransform(self.projection, resolution)

    def start_pan(self, x, y, button):
        """
        Called when a pan operation has started.

        *x*, *y* are the mouse coordinates in display coords.
        button is the mouse button number:

        * 1: LEFT
        * 2: MIDDLE
        * 3: RIGHT

        .. note::
            Intended to be overridden by new projection types.
        """
        self._pan_start = cbook.Bunch(
            lim           = self.viewLim.frozen(),
            trans         = self.transData.frozen(),
            trans_inverse = self.transData.inverted().frozen(),
            bbox          = self.bbox.frozen(),
            x             = x,
            y             = y,
            lastx         = x,
            lasty         = y,
            )

    def end_pan(self):
        """
        Called when a pan operation completes (when the mouse button
        is up.)

        .. note::
            Intended to be overridden by new projection types.
        """
        del self._pan_start

    def drag_pan(self, button, key, x, y):
        """
        Called when the mouse moves during a pan operation.

        *button* is the mouse button number:

        * 1: LEFT
        * 2: MIDDLE
        * 3: RIGHT

        *key* is a "shift" key

        *x*, *y* are the mouse coordinates in display coords.

        .. note::
            Intended to be overridden by new projection types.
        """
        def format_deltas(key, dx, dy):
            if key=='control':
                if(abs(dx)>abs(dy)):
                    dy = dx
                else:
                    dx = dy
            elif key=='x':
                dy = 0
            elif key=='y':
                dx = 0
            elif key=='shift':
                if 2*abs(dx) < abs(dy):
                    dx=0
                elif 2*abs(dy) < abs(dx):
                    dy=0
                elif(abs(dx)>abs(dy)):
                    dy=dy/abs(dy)*abs(dx)
                else:
                    dx=dx/ab+s(dx)*abs(dy)
            return (dx,dy)

        p = self._pan_start
        dx = x - p.x
        dy = y - p.y
        if dx == 0 and dy == 0:
            return
        if button == 1:
            dx, dy = format_deltas(key, dx, dy)
            x0, y0 = self.transAxes.inverted().transform([x, y])
            x1, y1 = self.transAxes.inverted().transform([p.lastx, p.lasty])
            p.lastx = x
            p.lasty = y
            dx = x0 - x1
            dy = y0 - y1

            self.patch.get_path().vertices += [dx, dy]
