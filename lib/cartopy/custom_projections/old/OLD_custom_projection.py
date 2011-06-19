import math

import numpy as np
import numpy.ma as ma

import matplotlib
rcParams = matplotlib.rcParams
from matplotlib.axes import Axes
from matplotlib import cbook
from matplotlib.patches import Circle
from matplotlib.path import Path
import matplotlib.spines as mspines
import matplotlib.axis as maxis
from matplotlib.ticker import Formatter, Locator, NullLocator, FixedLocator, NullFormatter
from matplotlib.transforms import Affine2D, Affine2DBase, Bbox, \
    BboxTransformTo, IdentityTransform, Transform, TransformWrapper

class GeoAxes(Axes):
    """
    An abstract base class for geographic projections
    """
    class ThetaFormatter(Formatter):
        """
        Used to format the theta tick labels.  Converts the native
        unit of radians into degrees and adds a degree symbol.
        """
        def __init__(self, round_to=1.0):
            self._round_to = round_to

        def __call__(self, x, pos=None):
            degrees = (x / np.pi) * 180.0
            degrees = round(degrees / self._round_to) * self._round_to
            if rcParams['text.usetex'] and not rcParams['text.latex.unicode']:
                return r"$%0.0f^\circ$" % degrees
            else:
                return u"%0.0f\u00b0" % degrees

    RESOLUTION = 75

    def _init_axis(self):
        self.xaxis = maxis.XAxis(self)
        self.yaxis = maxis.YAxis(self)
        # Do not register xaxis or yaxis with spines -- as done in
        # Axes._init_axis() -- until GeoAxes.xaxis.cla() works.
        # self.spines['geo'].register_axis(self.yaxis)
        self._update_transScale()

    def cla(self):
        Axes.cla(self)

        self.set_longitude_grid(30)
        self.set_latitude_grid(15)
        self.set_longitude_grid_ends(75)
        self.xaxis.set_minor_locator(NullLocator())
        self.yaxis.set_minor_locator(NullLocator())
        self.xaxis.set_ticks_position('none')
        self.yaxis.set_ticks_position('none')
        self.yaxis.set_tick_params(label1On=True)
        # Why do we need to turn on yaxis tick labels, but
        # xaxis tick labels are already on?

        self.grid(rcParams['axes.grid'])

        Axes.set_xlim(self, -np.pi, np.pi)
        Axes.set_ylim(self, -np.pi / 2.0, np.pi / 2.0)

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

        # This is the transform for longitude ticks.
        self._xaxis_pretransform = \
            Affine2D() \
            .scale(1.0, self._latitude_cap * 2.0) \
            .translate(0.0, -self._latitude_cap)
        self._xaxis_transform = \
            self._xaxis_pretransform + \
            self.transData
        self._xaxis_text1_transform = \
            Affine2D().scale(1.0, 0.0) + \
            self.transData + \
            Affine2D().translate(0.0, 4.0)
        self._xaxis_text2_transform = \
            Affine2D().scale(1.0, 0.0) + \
            self.transData + \
            Affine2D().translate(0.0, -4.0)

        # This is the transform for latitude ticks.
        yaxis_stretch = Affine2D().scale(np.pi * 2.0, 1.0).translate(-np.pi, 0.0)
        yaxis_space = Affine2D().scale(1.0, 1.1)
        self._yaxis_transform = \
            yaxis_stretch + \
            self.transData
        yaxis_text_base = \
            yaxis_stretch + \
            self.transProjection + \
            (yaxis_space + \
             self.transAffine + \
             self.transAxes)
        self._yaxis_text1_transform = \
            yaxis_text_base + \
            Affine2D().translate(-8.0, 0.0)
        self._yaxis_text2_transform = \
            yaxis_text_base + \
            Affine2D().translate(8.0, 0.0)

    def _get_affine_transform(self):
        transform = self._get_core_transform(1)
        xscale, _ = transform.transform_point((np.pi, 0))
        _, yscale = transform.transform_point((0, np.pi / 2.0))
        return Affine2D() \
            .scale(0.5 / xscale, 0.5 / yscale) \
            .translate(0.5, 0.5)

    def get_xaxis_transform(self,which='grid'):
        assert which in ['tick1','tick2','grid']
        return self._xaxis_transform

    def get_xaxis_text1_transform(self, pad):
        return self._xaxis_text1_transform, 'bottom', 'center'

    def get_xaxis_text2_transform(self, pad):
        return self._xaxis_text2_transform, 'top', 'center'

    def get_yaxis_transform(self,which='grid'):
        assert which in ['tick1','tick2','grid']
        return self._yaxis_transform

    def get_yaxis_text1_transform(self, pad):
        return self._yaxis_text1_transform, 'center', 'right'

    def get_yaxis_text2_transform(self, pad):
        return self._yaxis_text2_transform, 'center', 'left'

    def _gen_axes_patch(self):
        return Circle((0.5, 0.5), 0.5, transform=self.transAxes)

    def _gen_axes_spines(self):
        return {'geo':mspines.Spine.circular_spine(self,
                                                   (0.5, 0.5), 0.5)}
        
    def set_yscale(self, *args, **kwargs):
        if args[0] != 'linear':
            raise NotImplementedError

    set_xscale = set_yscale

    def set_xlim(self, *args, **kwargs):
        Axes.set_xlim(self, -np.pi, np.pi)
        Axes.set_ylim(self, -np.pi / 2.0, np.pi / 2.0)

    set_ylim = set_xlim

    def format_coord(self, long, lat):
        'return a format string formatting the coordinate'
        long = long * (180.0 / np.pi)
        lat = lat * (180.0 / np.pi)
        if lat >= 0.0:
            ns = 'N'
        else:
            ns = 'S'
        if long >= 0.0:
            ew = 'E'
        else:
            ew = 'W'
        return u'%f\u00b0%s, %f\u00b0%s' % (abs(lat), ns, abs(long), ew)

    def set_longitude_grid(self, degrees):
        """
        Set the number of degrees between each longitude grid.
        """
        number = (360.0 / degrees) + 1
        self.xaxis.set_major_locator(
            FixedLocator(
                np.linspace(-np.pi, np.pi, number, True)[1:-1]))
        self._logitude_degrees = degrees
        self.xaxis.set_major_formatter(self.ThetaFormatter(degrees))

    def set_latitude_grid(self, degrees):
        """
        Set the number of degrees between each longitude grid.
        """
        number = (180.0 / degrees) + 1
        self.yaxis.set_major_locator(
            FixedLocator(
                np.linspace(-np.pi / 2.0, np.pi / 2.0, number, True)[1:-1]))
        self._latitude_degrees = degrees
        self.yaxis.set_major_formatter(self.ThetaFormatter(degrees))

    def set_longitude_grid_ends(self, degrees):
        """
        Set the latitude(s) at which to stop drawing the longitude grids.
        """
        self._latitude_cap = degrees * (np.pi / 180.0)
        self._xaxis_pretransform \
            .clear() \
            .scale(1.0, self._latitude_cap * 2.0) \
            .translate(0.0, -self._latitude_cap)

    def get_data_ratio(self):
        '''
        Return the aspect ratio of the data itself.
        '''
        return 1.0

    ### Interactive panning

    def can_zoom(self):
        """
        Return True if this axes support the zoom box
        """
        return False

    def start_pan(self, x, y, button):
        pass

    def end_pan(self):
        pass

    def drag_pan(self, button, key, x, y):
        pass


class MollweideAxes(GeoAxes):
    name = 'mollweide2'

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
        
        def __init__(self, resolution):
            """
            Create a new Mollweide transform.  Resolution is the number of steps
            to interpolate between each input line segment to approximate its
            path in curved Mollweide space.
            """
            Transform.__init__(self)
            self._resolution = resolution

        def transform_lon_lat(self, longitude, latitude):
            def d(theta):
                delta = -(theta + np.sin(theta) - pi_sin_l) / (1 + np.cos(theta))
                return delta, abs(delta) > 0.001

            pi_sin_l = np.pi * np.sin(latitude)
            theta = 2.0 * latitude
            delta, large_delta = d(theta)
            i=0
            while np.any(large_delta):
                i+= 1
                theta += np.where(large_delta, delta, 0)
                delta, large_delta = d(theta)
            aux = theta / 2

            x = (2.0 * np.sqrt(2.0) * longitude * np.cos(aux)) / np.pi
            y = (np.sqrt(2.0) * np.sin(aux))

            return x, y 

        def transform(self, ll):
            longitude = ll[:, 0:1]
            latitude  = ll[:, 1:2]
            
            longitude = longitude.copy()
            # Scale the range
            longitude += np.pi
            longitude %= 2 * np.pi
            longitude -= np.pi
            
            x, y = self.transform_lon_lat(longitude, latitude)
            
            return np.concatenate([x, y], 1)

        def transform_no_mod(self, ll):
            longitude = ll[:, 0:1]
            latitude  = ll[:, 1:2]
            return np.concatenate(self.transform_lon_lat(longitude, latitude), 1)

        def transform_affine(self, values):
            print 'affine'
            return Transform.transform_affine(self, values)

        def transform_non_affine(self, points):
            print 'non affine'
            r = self.transform(points)
            res = np.where(r > np.pi)[0]
            if len(res) != 0:
                r[res] -= np.pi*2
            return r
        transform_non_affine.__doc__ = Transform.transform_non_affine.__doc__

        def transform_point(self, point):
            x, y = self.transform_lon_lat(point[0], point[1])
            if x > np.pi:
                x -= np.pi*2
            return x, y

        def transform_path(self, path):
            vertices = path.vertices
            # Intelligent interpolation needed
#            ipath = path.interpolated(self._resolution)
            ipath = path
            ipath = path.interpolated(10)
#            ipath = path.interpolated(50)
            
            verts = self.transform_no_mod(ipath.vertices)
            codes = ipath.codes
#            print 'transforming lon range:', np.min(verts[:, 0]), np.max(verts[:, 0])
#            if np.isnan(np.max(verts[:, 0])):
#                print 'Got nan: ', path, verts
                
            paths = []
            paths.append(Path(verts, codes))
            
            # Have any of the points wrapped? If so, pick up the pen, and start from -360
            if any(ipath.vertices[:, 0] > np.pi):
                v = ipath.vertices.copy()
#                print 'splitting -:'
                v[:, 0] -= 2 * np.pi
#                print v
                v = self.transform_no_mod(v)
                paths.append(Path(v))
                 
            # Have any of the points wrapped? If so, pick up the pen, and start from +360
            if any(ipath.vertices[:, 0] < -np.pi):
                v = ipath.vertices.copy()
                v[:, 0] += 2 * np.pi
                v = self.transform_no_mod(v)
                paths.append(Path(v))
                                                  
            if len(paths) == 1:
                path = paths[0]
            else:
                for path in paths:
                    if path.codes is not None:
                        if path.codes[0] == Path.MOVETO and all(path.codes[1:] == Path.LINETO):
                            path.codes = None
                        else:
                            # This is a bit strict... but a condition of make_compound_path
                            raise ValueError('Cannot draw discontiguous polygons.')
#                        print path.codes
                path = Path.make_compound_path(*paths)
            return path
        
        transform_path.__doc__ = Transform.transform_path.__doc__

        def transform_path_non_affine(self, path):
            print 'non affine path'

        def inverted(self):
            return MollweideAxes.InvertedMollweideTransform(self._resolution)
        inverted.__doc__ = Transform.inverted.__doc__

    class InvertedMollweideTransform(Transform):
        input_dims = 2
        output_dims = 2
        is_separable = False

        def __init__(self, resolution):
            Transform.__init__(self)
            self._resolution = resolution

        def transform(self, xy):
            # MGDTODO: Math is hard ;(
            return xy
        transform.__doc__ = Transform.transform.__doc__

        def inverted(self):
            return MollweideAxes.MollweideTransform(self._resolution)
        inverted.__doc__ = Transform.inverted.__doc__

    def __init__(self, *args, **kwargs):
        self._latitude_cap = np.pi / 2.0
        GeoAxes.__init__(self, *args, **kwargs)
        self.set_aspect(0.5, adjustable='box', anchor='C')
        self.cla()

    def _get_core_transform(self, resolution):
        return self.MollweideTransform(resolution)


class LambertAxes(GeoAxes):
    name = 'lambert2'

    class LambertTransform(Transform):
        """
        The base Lambert transform.
        """
        input_dims = 2
        output_dims = 2
        is_separable = False

        def __init__(self, center_longitude, center_latitude, resolution):
            """
            Create a new Lambert transform.  Resolution is the number of steps
            to interpolate between each input line segment to approximate its
            path in curved Lambert space.
            """
            Transform.__init__(self)
            self._resolution = resolution
            self._center_longitude = center_longitude
            self._center_latitude = center_latitude


        def transform(self, ll):
            longitude = ll[:, 0:1]
            latitude  = ll[:, 1:2]
            
            longitude = longitude.copy()
            # Scale the range
            longitude += np.pi
            longitude %= 2 * np.pi
            longitude -= np.pi
            
            x, y = self.transform_lon_lat(longitude, latitude)
            
            return np.concatenate([x, y], 1)

        def transform_no_mod(self, ll):
            longitude = ll[:, 0:1]
            latitude  = ll[:, 1:2]
            return np.concatenate(self.transform_lon_lat(longitude, latitude), 1)

        def transform_affine(self, values):
            print 'affine'
            return Transform.transform_affine(self, values)

        def transform_non_affine(self, points):
            print 'non affine'
            r = self.transform(points)
            res = np.where(r > np.pi)[0]
            if len(res) != 0:
                r[res] -= np.pi*2
            return r
        transform_non_affine.__doc__ = Transform.transform_non_affine.__doc__

        def transform_point(self, point):
            x, y = self.transform_lon_lat(point[0], point[1])
            if x > np.pi:
                x -= np.pi*2
            return x, y

        def transform_path(self, path):
            vertices = path.vertices
            # Intelligent interpolation needed
#            ipath = path.interpolated(self._resolution)
            ipath = path
            ipath = path.interpolated(10)
#            ipath = path.interpolated(3050)
            
            verts = self.transform_no_mod(ipath.vertices)
            codes = ipath.codes
#            print verts.shape
#            print 'transforming lon range:', np.min(verts[:, 0]), np.max(verts[:, 0])
#            if np.isnan(np.max(verts[:, 0])):
#                print 'Got nan: ', path, verts
                 
            paths = []
            paths.append(Path(verts, codes))
            
#            # Have any of the points wrapped? If so, pick up the pen, and start from -360
#            if any(ipath.vertices[:, 0] > np.pi):
#                v = ipath.vertices.copy()
#                print 'splitting -:'
#                v[:, 0] -= 2 * np.pi                
#                print v
#                v = self.transform_no_mod(v)
#                paths.append(Path(v))
#                 
#            # Have any of the points wrapped? If so, pick up the pen, and start from +360
#            if any(ipath.vertices[:, 0] < -np.pi):
#                v = ipath.vertices.copy()
#                v[:, 0] += 2 * np.pi
#                print 'splitting +:'
#                v = self.transform_no_mod(v)
#                paths.append(Path(v))
            s_pole = np.deg2rad(np.array([0, -89.9999]))
            if path.contains_point(s_pole):
                print 'POLE ALERT!!!', path 
                path = Path(verts[:-31])
                paths = [path]
            
            if len(paths) == 1:
                path = paths[0]
            else:
                for path in paths:
                    if path.codes is not None:
                        if path.codes[0] == Path.MOVETO and all(path.codes[1:] == Path.LINETO):
                            path.codes = None
                        else:
                            # This is a bit strict... but a condition of make_compound_path
                            raise ValueError('Cannot draw discontiguous polygons.')
#                        print path.codes
                path = Path.make_compound_path(*paths)
                                      
            return path
        
        transform_path.__doc__ = Transform.transform_path.__doc__

        def transform_path_non_affine(self, path):
            print 'non affine path'


        def transform_lon_lat(self, longitude, latitude):
            clong = self._center_longitude
            clat = self._center_latitude
            cos_lat = np.cos(latitude)
            sin_lat = np.sin(latitude)
            diff_long = longitude - clong
            cos_diff_long = np.cos(diff_long)

            inner_k = (1.0 +
                       np.sin(clat)*sin_lat +
                       np.cos(clat)*cos_lat*cos_diff_long)
            # Prevent divide-by-zero problems
            inner_k = np.where(inner_k == 0.0, 1e-15, inner_k)
            k = np.sqrt(2.0 / inner_k)
            x = k*cos_lat*np.sin(diff_long)
            y = k*(np.cos(clat)*sin_lat -
                   np.sin(clat)*cos_lat*cos_diff_long)

            return x, y
        
        def inverted(self):
            return LambertAxes.InvertedLambertTransform(
                self._center_longitude,
                self._center_latitude,
                self._resolution)
        inverted.__doc__ = Transform.inverted.__doc__


    class InvertedLambertTransform(Transform):
        input_dims = 2
        output_dims = 2
        is_separable = False

        def __init__(self, center_longitude, center_latitude, resolution):
            Transform.__init__(self)
            self._resolution = resolution
            self._center_longitude = center_longitude
            self._center_latitude = center_latitude

        def transform(self, xy):
            x = xy[:, 0:1]
            y = xy[:, 1:2]
            clong = self._center_longitude
            clat = self._center_latitude
            p = np.sqrt(x*x + y*y)
            p = np.where(p == 0.0, 1e-9, p)
            c = 2.0 * np.arcsin(0.5 * p)
            sin_c = np.sin(c)
            cos_c = np.cos(c)

            lat = np.arcsin(cos_c*np.sin(clat) +
                             ((y*sin_c*np.cos(clat)) / p))
            long = clong + np.arctan(
                (x*sin_c) / (p*np.cos(clat)*cos_c - y*np.sin(clat)*sin_c))

            return np.concatenate((long, lat), 1)
        transform.__doc__ = Transform.transform.__doc__

        def inverted(self):
            return LambertAxes.LambertTransform(
                self._center_longitude,
                self._center_latitude,
                self._resolution)
        inverted.__doc__ = Transform.inverted.__doc__

    def __init__(self, *args, **kwargs):
        self._latitude_cap = np.pi / 2.0
        self._center_longitude = kwargs.pop("center_longitude", 0.0)
        self._center_latitude = kwargs.pop("center_latitude", 0.0)
        GeoAxes.__init__(self, *args, **kwargs)
        self.set_aspect('equal', adjustable='box', anchor='C')
        self.cla()

    def cla(self):
        GeoAxes.cla(self)
        self.yaxis.set_major_formatter(NullFormatter())

    def _get_core_transform(self, resolution):
        return self.LambertTransform(
            self._center_longitude,
            self._center_latitude,
            resolution)

    def _get_affine_transform(self):
        return Affine2D() \
            .scale(0.25) \
            .translate(0.5, 0.5)

from matplotlib.projections import register_projection


# Now register the projection with matplotlib so the user can select
# it.
register_projection(MollweideAxes)
register_projection(LambertAxes)



if __name__ == '__main__':
    import custom_projection
    import matplotlib.pyplot as plt
    import numpy as np
    
#    plt.subplot(111, projection="lambert2")
    plt.subplot(111, projection="mollweide2")
#    p = plt.plot([-1, 1, 1, 2*np.pi - 0.5, -1 ], [1, -1, 1, -1.3, 1], "o-")
#    p = plt.plot([0, 8*np.pi, ], [-np.pi/4, np.pi/4], "o-")


    import matplotlib
    from matplotlib.collections import PatchCollection
    import matplotlib.patches as mpatches
    
##    plt.subplot(111, projection="mollweide2")

#    poly = mpatches.RegularPolygon( (np.pi, 0), 5, 1.0)
#    collection = PatchCollection([poly], cmap=matplotlib.cm.jet, alpha=0.4)
#    plt.gca().add_collection(collection)
#    
#    # make up some data on a regular lat/lon grid.
#    lons = np.linspace(0, 2 * np.pi, 145)
#    lats = np.linspace(-np.pi/2, np.pi/2, 74)
#    lons, lats = np.meshgrid(lons, lats)
#    
#    wave = 0.75*(np.sin(2.*lats)**8*np.cos(4.*lons))
#    mean = 0.5*np.cos(2.*lats)*((np.sin(2.*lats))**2 + 2.)
#    
#    print 'doing contour'
    
    #CS = plt.contour(lons, lats, wave+mean, 15, linewidths=1.5)
#    CS = plt.contourf(lons, lats, wave+mean, 15, alpha=0.5)
    
    plt.grid()
    
    plt.show()
    
    
    
