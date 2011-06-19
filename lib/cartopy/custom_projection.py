import math
import numpy as np
import numpy.ma as ma

import matplotlib
from mpl_toolkits.mplot3d.art3d import mpath
from shapely.geometry.collection import GeometryCollection
from shapely.geometry.multilinestring import MultiLineString
import shapely.geometry
from shapely.ops import cascaded_union


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


import cartopy.coastlines
import cartopy.geos_to_mpl_path_conversion as shape_convert

class ThetaFormatter(Formatter):
        """
        Used to format the theta tick labels.  Converts the native
        unit of radians into degrees and adds a degree symbol.
        """
        def __call__(self, degrees, pos=None):
            if rcParams['text.usetex'] and not rcParams['text.latex.unicode']:
                return r"$%0.0f^\circ$" % degrees
            else:
                return u"%0.0f\u00b0" % degrees


class GeoAxes(Axes):
    """
    An abstract base class for geographic projections
    """
    
    RESOLUTION = 75

    def _init_axis(self):
        self.xaxis = maxis.XAxis(self)
        self.yaxis = maxis.YAxis(self)
        # Do not register xaxis or yaxis with spines -- as done in
        # Axes._init_axis() -- until GeoAxes.xaxis.cla() works.
        # self.spines['geo'].register_axis(self.yaxis)
        self._update_transScale()

    def coastlines(self):
        cartopy.coastlines.add_coastlines()

    def cla(self):
        Axes.cla(self)

        self.set_longitude_grid(30)
        self.set_latitude_grid(10)
        self.set_longitude_grid_ends(89)
        self.set_latitude_grid_ends(89)
        self.xaxis.set_minor_locator(NullLocator())
        self.yaxis.set_minor_locator(NullLocator())
        self.xaxis.set_ticks_position('none')
        self.yaxis.set_ticks_position('none')
#        self.yaxis.set_tick_params(label1On=True)
        # Why do we need to turn on yaxis tick labels, but
        # xaxis tick labels are already on?

        self.grid(rcParams['axes.grid'])

        Axes.set_xlim(self, -180, 179.999)
#        Axes.set_ylim(self, -90, 90)
        Axes.set_ylim(self, -80, 80)
        
    def _set_lim_and_transforms(self):
        # A (possibly non-linear) projection on the (already scaled) data
        self.transProjection = self._get_core_transform(self.RESOLUTION)

        self.transAffine = self._get_affine_transform()

        self.transAxes = BboxTransformTo(self.bbox)

        # The complete data transformation stack -- from data all the
        # way to display coordinates
        self.transData = \
            self.transProjection + \
            self.transAffine + \
            self.transAxes

        self._latitude_axis_longitude = 30

        # This is the transform for longitude ticks.
        self._xaxis_pretransform = \
            Affine2D() \
            .scale(1.0, self._latitude_cap * 2.0) \
            .translate(0.0, -self._latitude_cap)
        self._xaxis_transform = \
            self._xaxis_pretransform + \
            self.transData
        self._xaxis_text1_transform = Affine2D().translate(0, self._latitude_axis_longitude) + self.transData
        self._xaxis_text2_transform = Affine2D().translate(0, self._latitude_axis_longitude) + self.transData
                                      
        # This is the transform for r-axis ticks.  It scales the theta
        # axis so the gridlines from 0.0 to 1.0, now go from 0.0 to
        # 2pi.
        
#        # This is the transform for longitude ticks.
        self._yaxis_pretransform = \
            Affine2D() \
            .scale(1.0, 89 * 2.0) \
            .translate(0.0, -89)
        self._yaxis_transform = \
            self._yaxis_pretransform + \
            self.transData

        yaxis_stretch = Affine2D().scale(360, 1.0).translate(-180, 0.0)
        
        self._yaxis_transform = yaxis_stretch + self.transData
#        self._yaxis_transform = self.transData
        # The r-axis labels are put at an angle and padded in the r-direction
        self._rpad = 0.05
        
        self._longitude_axis_latitude = -80
        self._r_label1_position = Affine2D().translate(self._longitude_axis_latitude, self._rpad)
        self._yaxis_text1_transform = self._r_label1_position + self.transData
        
        self._r_label2_position = Affine2D().translate(self._longitude_axis_latitude, self._rpad)
        self._yaxis_text2_transform = self._r_label2_position + self.transData

    def _get_affine_transform(self):
        # The scaling transform to go from proj XY meters to axes coords
        transform = self._get_core_transform(1)
        
#        x_extreme, y_extreme = self.projection.get_extreme_points()
#        xscale, _ = transform.transform_point(x_extreme)
#        _, yscale = transform.transform_point(y_extreme)
#        return Affine2D().scale(0.5 / xscale, 0.5 / yscale).translate(0.5, 0.5)

        
        x_extreme, y_extreme = self.projection.forward(self.projection.edge.interpolated(100).vertices[:, 0], 
                                                       self.projection.edge.interpolated(100).vertices[:, 1], nans=True)
#        print x_extreme, y_extreme
        xscale, yscale = np.nanmax(np.abs(x_extreme)), np.nanmax(np.abs(y_extreme))

#        print 'scales: ', xscale, yscale
        return Affine2D().scale(0.5 / xscale, 0.5 / yscale).translate(0.5, 0.5)

    def get_xaxis_transform(self,which='grid'):
        assert which in ['tick1','tick2','grid']
        return self._xaxis_transform

    def get_xaxis_text1_transform(self, pad):
        return self._xaxis_text1_transform, 'bottom', 'center'

    def get_xaxis_text2_transform(self, pad):
        return self._xaxis_text2_transform, 'top', 'center'

    def get_yaxis_transform(self, which='grid'):
        assert which in ['tick1','tick2','grid']
        return self._yaxis_transform

    def get_yaxis_text1_transform(self, pad):
        return self._yaxis_text1_transform, 'center', 'right'

    def get_yaxis_text2_transform(self, pad):
        return self._yaxis_text2_transform, 'center', 'left'

    def _gen_axes_patch(self):
        import matplotlib.patches as mpatches
        s = self.projection.edge
        p = self.transData.transform_path(s)
        p = self.transAxes.inverted().transform_path(p)
        return mpatches.PathPatch(p, transform=self.transAxes)
#        return mpatches.Rectangle([0,0], 1, 1, transform=self.transAxes)
        return Circle((0.5, 0.5), 0.5, transform=self.transAxes)

    def _gen_axes_spines(self):
        return {}
    
    def set_yscale(self, *args, **kwargs):
        if args[0] != 'linear':
            raise NotImplementedError

    set_xscale = set_yscale

    def set_xlim(self, *args, **kwargs):
        Axes.set_xlim(self, -180, 180)
        Axes.set_ylim(self, -90, 90)

    set_ylim = set_xlim

    def format_coord(self, long, lat):
        """"return a format string formatting the coordinate."""
        ns = 'N' if lat >= 0.0 else 'S'
        ew = 'E' if long >= 0.0 else 'W'
        
        # XXX adaptive resolution
        return u'%f\u00b0%s, %f\u00b0%s' % (abs(lat), ns, abs(long), ew)
        
    def set_longitude_grid(self, degrees):
        """
        Set the number of degrees between each longitude grid.
        """
        self.xaxis.set_major_locator(LinearLocator(6))
        self.xaxis.set_major_formatter(ThetaFormatter())

    def set_latitude_grid(self, degrees):
        """
        Set the number of degrees between each longitude grid.
        """
        self.yaxis.set_major_locator(FixedLocator(np.linspace(-90, 90, degrees)[1:-1]))
        self.yaxis.set_major_formatter(ThetaFormatter())

    def set_longitude_grid_ends(self, degrees):
        """
        Set the latitude(s) at which to stop drawing the longitude grids.
        """
        self._latitude_cap = degrees
        self._xaxis_pretransform \
            .clear() \
            .scale(1.0, self._latitude_cap * 2.0) \
            .translate(0.0, -self._latitude_cap)

    def set_latitude_grid_ends(self, degrees):
        """
        Set the longitude(s) at which to stop drawing the latitude grids.
        """
        return
        self._latitude_cap = degrees
        self._yaxis_pretransform \
            .clear() \
            .scale(1.0, self._latitude_cap * 2.0) \
            .translate(0.0, -self._latitude_cap)

    def get_data_ratio(self):
        '''
        Return the aspect ratio of the data itself.
        '''
        return 1.0

    def start_pan(self, x, y, button):
        pass

    def end_pan(self):
        pass

    def drag_pan(self, button, key, x, y):
        pass


class LongitudeWrappingTransform(Transform):
    """
    Represents projections, such a cylindrical & conic who have a line which neighbours another
    line in a disjoint way. For example, the dateline on a standard Cylindrial plot is the cutline
    between the left and right hand side of the plot  
    """
    input_dims = 2
    output_dims = 2
    is_separable = False

    def __str__(self):
        return repr(self)
    
    def __init__(self, projection, resolution):
        self.projection = projection
        Transform.__init__(self)
      
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
        
#        p2 = polygon.Polygon(((-180, -90), (180, -90), (180, 90), (-180, 90)))
        p2 = self.projection.edge
        p2 = path_interpolation(p2, 30)
        p2 = shape_convert.path_to_geos(p2)
        p2 = p2[0]
        
        paths = []
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
            
            # Turn the list of shapes into a Geos type    
            if isinstance(shp, polygon.Polygon):
                shps = MultiPolygon(shps)
            elif isinstance(shp, linestring.LineString):
                shps = MultiLineString(shps)
            else:
                ValueError('Unknown shape type')
            
            # join the shapes back together again if they have wrapped the entire 360 range.
            shps = cascaded_union(shps)        
            
            # Do one more intersection of the overall union (this seems to be necessary)
            intersection_shp = shps.intersection(p2)
            pths = shape_convert.geos_to_path(intersection_shp)

#            pths = shape_convert.geos_to_path(shps)
            
            # Now interpolate the paths 
            interp_resolution = 15
#            interp_resolution = 5
    
            
#            for pth in pths:
#                p9 = path_interpolation(pth, 9)
##                print '\n------' * 30
##                print 'interp 9:', p9
##                print '\n------' * 5
##                print 'interp 5:', path_interpolation(pth, 5)
##                print '\n------' * 30
#                paths.append(path_interpolation(pth, 9))
#            
            paths.extend([path_interpolation(pth, interp_resolution) for pth in pths])
    
    
        if len(paths) == 1:
            path = paths[0]
        elif len(paths) == 0:
            return Path(np.empty([0,2]))
        else:
            points = []
            codes = []
            for path in paths:
                path_points, path_codes = zip(*(path.iter_segments(curves=False, simplify=False)))
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
        path.vertices = self.transform_no_mod(path.vertices)

        return path

    def inverted(self):
        return InvertedLongitudeWrappingTransform(self.projection)


class InvertedLongitudeWrappingTransform(Transform):
    input_dims = 2
    output_dims = 2
    is_separable = False

    def __init__(self, projection):
        self.projection = projection
        Transform.__init__(self)

    def transform_x_y(self, x, y):
        return self.projection.inverse(x, y, radians=False)
    
    def transform(self, xy):
        lon, lat = self.transform_x_y(xy[:, 0], xy[:, 1])
        r = np.concatenate([lon, lat], 1)
        r.shape = xy.shape 
        
        return r    

    def inverted(self):
        return LongitudeWrappingTransform(self._resolution)


class WrappedGeoAxes(GeoAxes):
#     is this really a cylindrical projection class? Does this also represent conical projections.
    def __init__(self, projection, *args, **kwargs):
        self.projection = projection
        self._latitude_cap = np.pi / 2.0
        GeoAxes.__init__(self, *args, **kwargs)
        self.set_aspect(0.5, adjustable='box', anchor='C')
        self.cla()

    def __str__(self):
        return 'WrappedGeoAxes with map projection: %s' % (self.projection, )
    
    def _get_core_transform(self, resolution):
        return LongitudeWrappingTransform(self.projection, resolution)

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


class LambertTransform(LongitudeWrappingTransform):
    def cut_path(self, path):
        rpaths = [path]
        
        # check point intersections at 180, 0 & 0, 45
        if path.intersects_path(mpath.Path([[180, 0], [180, 0]])):
            fix_points = []
            import custom_projections.path_intersections
            for i, p1 in enumerate(path.vertices[1:, :]):
                p2 = path.vertices[i, :]
                # This should not include those points which are exactly [180, 0]
                intersect = custom_projections.path_intersections.intersecton_point(p1[0], p2[0], 180, 179.9999,
                                                                    p1[1], p2[1], 0, 0.0001)
                 
                if intersect is not None:
                    fix_points.append(i+1)
                    
            if not fix_points:
                RuntimeError('No intersection found, but Matplotlib said there should be.')

            rpaths2 = []
            print 'fix points:', fix_points, zip(fix_points, [None, fix_points[1:]]),
            
            
            rpaths = [mpath.Path(verts) for verts in np.split(path.vertices, fix_points)]
            print 'splitted: ', rpaths 
            
            for a in rpaths[:-1]:
                a.vertices = np.concatenate([a.vertices, [[180, -0.001]]])
            
            for a in rpaths[1:]:
                a.vertices = np.concatenate([[[180, 0.001]], a.vertices])
             
#            for i in fix_points:
#            for i, j in zip([0, fix_points], [fix_points[:], None]):                
#                print path.vertices[i, :], path.vertices[i+1, :], path.vertices[i:j, :]
#                rpaths2.append(mpath.Path())
        return rpaths
    
    def transform_path(self, path):
        print 'transform called: '
        r = self.cut_path(path)
        print r
        print path
        path = r[0]
        path.vertices = self.transform_no_mod(path.vertices)
        return path.interpolated(100)
        return path
        
    
class LambertAxes(WrappedGeoAxes):
#    def __init__(self, projection, *args, **kwargs):
#        self.projection = projection
#        self._latitude_cap = 90
#        GeoAxes.__init__(self, *args, **kwargs)
#        self.set_aspect(0.5, adjustable='box', anchor='C')
#        self.cla()
        
        # ONLY HAS 1 CUT POINT - no edge....

    def _get_core_transform(self, resolution):
        return LambertTransform(self.projection, resolution)

        
    def _get_affine_transform(self):
        # the transform which goes from data to axes coordinates.... I THINK
        transform = self._get_core_transform(1)
        
        # the two extremes of the lambert axes are very close to one another...
        xscale, _ = transform.transform_point((179.999, 0.0))
        _, yscale = transform.transform_point((180, 0.001))
        
        return Affine2D().scale(0.5 / xscale, 0.5 / yscale).translate(0.5, 0.5)

    def _gen_axes_patch(self):
        return Circle((0.5, 0.5), 0.5, transform=self.transAxes)


class EAAxes(WrappedGeoAxes):
    def __init__(self, *args, **kwargs):
        import projections
        self.projection = projections.EquiRectangular()
        self._latitude_cap = 90
        GeoAxes.__init__(self, *args, **kwargs)
        self.set_aspect(0.5, adjustable='box', anchor='C')
        self.cla()

