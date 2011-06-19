import pyproj as proj
import numpy as np

import matplotlib.axes
import matplotlib.path as mpath


import cartopy.custom_projection

# some statuses:
NotWorking = 1
PartiallyWorking = 1
FullyFunctional_no_kwargs = 1
FullyFunctional = 1


def NoInverseMethod(self, *args, **kwargs):
        raise NotImplementedError('This projection does not have an inverse.')



class CartopySubplot(matplotlib.axes.SubplotBase, cartopy.custom_projection.WrappedGeoAxes):
    def __init__(self, projection, *args, **kwargs):
        self.projection = projection
        matplotlib.axes.SubplotBase.__init__(self, *args, **kwargs)
        
    def _init_axes(self, fig, **kwargs):
        # _axes_class is set in the subplot_class_factory
        cartopy.custom_projection.WrappedGeoAxes.__init__(self, self.projection, fig, self.figbox, **kwargs)


class CartopyLambertSubplot(matplotlib.axes.SubplotBase, cartopy.custom_projection.LambertAxes):
    def __init__(self, projection, *args, **kwargs):
        self.projection = projection
        matplotlib.axes.SubplotBase.__init__(self, *args, **kwargs)
        
    def _init_axes(self, fig, **kwargs):
        # _axes_class is set in the subplot_class_factory
        cartopy.custom_projection.LambertAxes.__init__(self, self.projection, fig, self.figbox, **kwargs)
        
        
class CartopyProjection(object):
    def subplot(self, *args, **kwargs):
        return  CartopySubplot(self, *args, **kwargs)
    
    def _repr_full_import(self):
        # Make this generic
        return 'cartopy.projections.%s()' % self.__class__.__name__
    
    def get_extreme_points(self):
        return (180, 0), (0, 90)
    
    def forward(self, lons, lats, radians=False, nans=False):
        """Converts lons & lats into a uv form (which can be plotted)."""
        x, y = self.proj(lons, lats, radians=radians)
        if nans:
            x = np.array(x)
            y = np.array(y)
    
            out_of_range = np.where((x == 1e30) | (y == 1e30))
            x[out_of_range] = y[out_of_range] = np.NaN
        
        return x, y

    def inverse(self, x, y, radians=False):
        """inverse of transform."""
        lon, lat = self.proj(x, y, radians=radians, inverse=True)
        return lon, lat

    
class Cylindrical(CartopyProjection):
    """Abstract """
    edge = mpath.Path(np.array([[-180, 180, 180, -180, -180], 
                                [-90, -90, 90, 90, -90]]).T
                               )


class Pseudocylindrical(CartopyProjection):
    """Abstract """
    edge = mpath.Path(np.array([[-180, 180, 180, -180, -180], 
                                            [-90, -90, 90, 90, -90]]).T
                               )


class ConicProjection(CartopyProjection):
    # XXX Fix inverted land
    edge = mpath.Path(np.array([[-180, 180, 180, -180, -180], 
                                [-90, -90, 90, 90, -90]]).T
                               )


class Pseudoconic(CartopyProjection):
    edge = mpath.Path(np.array([[-180, 180, 180, -180, -180], 
                                [-90, -90, 90, 90, -90]]).T
                               )



class TransverseProjection(CartopyProjection):
    pass


class Azimuthal(CartopyProjection):
    pass


class EquiRectangular(Cylindrical):
    """
    Equirectangular
Name     Equirectangular
Alias     Plate Caree
Alias     Equidistant Cylindrical
Alias     Simple Cylindrical
EPSG Code     9823 (spherical), 9842 (elliptical)
GeoTIFF Code     CT_Equirectangular (17)
OGC WKT Name     Equirectangular
Supported By     GeoTIFF, PROJ.4, OGC WKT
Projection Parameters
Name     EPSG #     GeoTIFF ID     OGC(OGR) WKT     ESRI PE WKT     PROJ.4     Units
Latitude of true scale     3     ProjStdParallel1     standard_parallel_1     Standard_Parallel_1     +lat_ts     Angular
Latitude of origin     1 (8801)     ProjCenterLat     latitude_of_origin     (unavailable)     +lat_0     Angular
Longitude of projection center     2 (8802/8822)     ProjCenterLong     central_meridian     Central_Meridian     +lon_0     Angular
False Easting     6     FalseEasting     false_easting     False_Easting     +x_0     Linear
False Northing     7     FalseNorthing     false_northing     False_Northing     +y_0     Linear
Notes

There are two latitudes that can be important for this projection. 
The latitude of true scale (lat_ts in PROJ.4, Standard_Parallel_1 in ESRI PE, PSEUDO_STD_PARALLEL_1 in OGR (1.6.0beta), unavailable in EPSG, 6th parm in GCTP) and the latitude of origin (lat_0 in PROJ.4, latitude_of_origin in OGR WKT, unavailable in ESRI PE, unavailable in GCTP). The latitude of origin is just an alternate origin and has no other effect on the equations. Often both are zero.

Generally speaking, if you only have one of these latitudes provided, it is likely the latitude of true scale.

For a period OGR used the name "latitude_of_origin" for the latitude of true scale, in error resulting in much confusion.

Most libraries only implement the spherical version this projection. The rules for deriving a spherical radius from the ellipsoid defined for a coordinate system vary between libraries, causing some confusion and mixed results.

Plate Caree is an alias, but generally implies that the latitude of true scale, and latitude of natural origin are zero. Simple Cylindrical is also an alias with the same assumptions.

    """
    status = FullyFunctional_no_kwargs
    def __init__(self, central_meridian=0, central_parallel=0):
        self.proj = proj.Proj(
                              proj='eqc',
                              ellps='WGS84',
                              lon_0=central_meridian,
                              lat_0=central_parallel,
                              )

    
class Mollweide(Pseudocylindrical):
    status = FullyFunctional_no_kwargs
    def __init__(self, central_meridian=0):
        self.proj = proj.Proj(
                              proj='moll',
                              ellps='WGS84',
                              lon_0=central_meridian,
                              )

        
class Hammer(Pseudocylindrical):
    status = FullyFunctional_no_kwargs
    def __init__(self, central_meridian=0):
        self.proj = proj.Proj(
                              proj='hammer',
                              ellps='WGS84',
                              lon_0=central_meridian,
                              )


class Geos(Azimuthal):
    status = NotWorking
    def __init__(self, satellite_height=35786):
        self.proj = proj.Proj(
                              proj='geos',
                              ellps='WGS84',
                              h=satellite_height,
                              )

        self.edge = mpath.Path(np.array([[-180, 180, 180, -180], 
                                            [-90, -90, 90, 90]]).T
                               )


class Robinson(Pseudocylindrical):
    status = FullyFunctional_no_kwargs
    def __init__(self):
        self.proj = proj.Proj(
                              proj='robin',
                              ellps='WGS84'
                              )


class Sinusoidal(Pseudocylindrical):
    status = FullyFunctional_no_kwargs
    def __init__(self):
        self.proj = proj.Proj(
                              proj='sinu',
                              ellps='WGS84'
                              )


class Mercator(Cylindrical):
    status = FullyFunctional_no_kwargs
    def __init__(self):
        self.proj = proj.Proj(
                              proj='merc',
                              ellps='WGS84'
                              )
        
        # XXX Calculate these automatically
        self.edge = mpath.Path(np.array([[-180, 180, 180, -180, -180], 
                                            [-80, -80, 80, 80, -80]]).T
                               )


class TransverseMercator(TransverseProjection):
    status = NotWorking
    def __init__(self):
        self.proj = proj.Proj(
                              proj='tmerc',
                              ellps='WGS84'
                              )
        
        # XXX Calculate these automatically
        self.edge = mpath.Path(np.array([[-90, 90, 90, -90, -90], 
                                            [-80, -80, 80, 80, -80]]).T
                               )
        
        self.edge = mpath.Path(np.array([[-80, 80, 80, -80, -80], 
                                            [-80, -80, 80, 80, -80]]).T
                               )
        
    def get_extreme_points(self):
        return (90, 0), (90, 90)


class GallStereographic(Cylindrical):
    status = FullyFunctional_no_kwargs
    def __init__(self):
        self.proj = proj.Proj(
                              proj='gall',
                              ellps='WGS84'
                              )


class Miller(Cylindrical):
    status = FullyFunctional_no_kwargs
    def __init__(self):
        self.proj = proj.Proj(
                              proj='mill',
                              ellps='WGS84'
                              )


class Polyconic(CartopyProjection):
    status = FullyFunctional_no_kwargs
    def __init__(self):
        self.proj = proj.Proj(
                              proj='poly',
                              ellps='WGS84'
                              )

        self.edge = mpath.Path(np.array([[-180, 180, 180, -180, -180], 
                                            [-90, -90, 90, 90, -90]]).T
                               )

class Stereographic(Azimuthal):
    status = PartiallyWorking
    # In particular, standard equatorial stereo is working fine.
    def __init__(self, lat_0=0):
        self.proj = proj.Proj(
                              proj='stere',
                              ellps='WGS84',
                              lat_0=lat_0,
                              )
        self.lat_0 = lat_0

        self.edge = mpath.Path(np.array([[-90, 90, 90, -90, -90], 
                                            [-90+lat_0, -90+lat_0, 90+lat_0, 90+lat_0, -90+lat_0]]).T
                               )
        
    def get_extreme_points(self):
        return (90, 0), (0, 90)
        
        
class EquidistantConic(ConicProjection):
    status = NotWorking
    def __init__(self):
        self.proj = proj.Proj(
                              proj='eqdc',
                              ellps='WGS84'
                              )

        self.edge = mpath.Path(np.array([[-180, 180, 180, -180, -180], 
                                            [-90, -90, 90, 90, -90]]).T
                               )


class Cassini(TransverseProjection):
    status = NotWorking
    def __init__(self):
        self.proj = proj.Proj(
                              proj='cass',
                              ellps='WGS84'
                              )

        # XXX This is not the correct edge: see http://en.wikipedia.org/wiki/Cassini_projection
        self.edge = mpath.Path(np.array([[-180, -90, -90, -180, -180], 
                                            [0, 0, -45, -45, -90]]).T
                               )
        
    def get_extreme_points(self):
        pts = (180, 0), (180, 90)
        p1, p2 = pts[0], pts[1]
        
        print 'p1', self.forward(p1[0], p1[1])
        print 'p2', self.forward(p2[0], p2[1])
        
        
        return pts 


class McBrydeThomasFlatPolarQuartic(CartopyProjection):
    status = FullyFunctional_no_kwargs
    def __init__(self):
        self.proj = proj.Proj(
                              proj='mbtfpq',
                              ellps='WGS84'
                              )
        self.edge = mpath.Path(np.array([[-180, 180, 180, -180, -180], 
                                            [-90, -90, 90, 90, -90]]).T
                               )


class Ortelius(CartopyProjection):
    status = FullyFunctional_no_kwargs
    def __init__(self):
        self.proj = proj.Proj(
                              proj='ortel',
                              ellps='WGS84'
                              )
        self.edge = mpath.Path(np.array([[-180, 180, 180, -180, -180], 
                                            [-90, -90, 90, 90, -90]]).T
                               )
        
    inverse = NoInverseMethod
    

class Lagrange(CartopyProjection):
    status = FullyFunctional_no_kwargs
    def __init__(self, lon_0=0, w=1.4):
        self.proj = proj.Proj(
                              proj='lagrng',
                              ellps='WGS84',
                              lon_0=lon_0,
                              W=1.4,
                              )
        self.edge = mpath.Path(np.array([[-180, 180, 180, -180, -180], 
                                            [-90, -90, 90, 90, -90]]).T
                               )

    inverse = NoInverseMethod


class AlbersEqualArea(ConicProjection):
    status = FullyFunctional_no_kwargs
    def __init__(self):
        self.proj = proj.Proj(
                              proj='aea',
                              ellps='WGS84'
                              )

        # XXX This is not the correct edge: see http://en.wikipedia.org/wiki/Cassini_projection
        self.edge = mpath.Path(np.array([[-180, 180, 180, -180, -180], 
                                            [-90, -90, 90, 90, -90]]).T
                               )

    def get_extreme_points(self):
        # XXX Figure these out. Can they be calculated based on the edge???
        return (180, 0), (144, 0)


class PerspectiveConic(ConicProjection):
    status = NotWorking
    def __init__(self, lat_1=20, lat_2=60):
        self.proj = proj.Proj(
                              proj='pconic',
                              ellps='WGS84',
                              lat_1 = lat_1,
                              lat_2 = lat_2,
                              )

        # XXX This is not the correct edge: see http://en.wikipedia.org/wiki/Cassini_projection
        self.edge = mpath.Path(np.array([[-180, 180, 180, -180, -180], 
                                            [20, 20, 80, 80, 20]]).T
                               )

    def get_extreme_points(self):
        # XXX Figure these out. Can they be calculated based on the edge???
        return (180, 0), (144, 0)


class VanderGrinten(CartopyProjection):
    status = FullyFunctional_no_kwargs

    def __init__(self, central_meridian=0.0):
        self.proj = proj.Proj(
                              proj='vandg',
                              ellps='WGS84',
                              lon_0=central_meridian,
                              )
        
        self.edge = mpath.Path(np.array([[-180, 180, 180, -180, -180], 
                                            [-90, -90, 90, 90, -90]]).T
                               )

    
class LambertConformalConic(ConicProjection):
    status = FullyFunctional_no_kwargs
    def __init__(self):
        self.proj = proj.Proj(
                              proj='lcc',
                              ellps='WGS84'
                              )

        self.edge = mpath.Path(np.array([[-180, 180, 180, -180, -180], 
                                            [-80, -80, 90, 90, -80]]).T
                               )
    
    def get_extreme_points(self):
        # XXX Extreme point calcualtion
        return (180, 10), (0, 90)
        

class LambertEqualArea(ConicProjection):
    status = FullyFunctional_no_kwargs
    def __init__(self):
        self.proj = proj.Proj(
                              proj='leac',
                              ellps='WGS84'
                              )

        self.edge = mpath.Path(np.array([[-180, 180, 180, -180, -180], 
                                            [-80, -80, 90, 90, -80]]).T
                               )
    
    def get_extreme_points(self):
        # XXX Extreme point calcualtion
        return (180, 10), (0, 90)


class Bonne(Pseudoconic):
    status = FullyFunctional_no_kwargs
    def __init__(self, lat_1=40):
        self.proj = proj.Proj(
                              proj='bonne',
                              ellps='WGS84', 
                              lat_1 = lat_1,
                              )

        self.edge = mpath.Path(np.array([[-180, 180, 180, -180, -180], 
                                            [-90, -90, 90, 90, -90]]).T
                               )

        
class Collignon(CartopyProjection):
    status = FullyFunctional_no_kwargs
    def __init__(self):
        self.proj = proj.Proj(
                              proj='collg',
                              ellps='WGS84'
                              )

        self.edge = mpath.Path(np.array([[-180, 180, 180, -180, -180], 
                                            [-80, -80, 90, 90, -80]]).T
                               )
    
    def get_extreme_points(self):
        return (180, 0), (0, 90)

        
class LambertAzimuthal(CartopyProjection):
    status = NotWorking
    # LAMBERTAZIMEQU
    def __init__(self, central_meridian=0):
        self.proj = proj.Proj(
                              proj='laea',
                              ellps='WGS84',
                              lon_0=central_meridian,
                              )
        import numpy
        import matplotlib.path as mpath
        self.edge = mpath.Path(numpy.array([[-170, 170, 170, -170, -170], 
                                            [-80, -80, 80, 80, -80]]).T
                               )
        
    def subplot(self, *args, **kwargs):
        return  CartopyLambertSubplot(self, *args, **kwargs)
                


if __name__ == '__main__':
    import cartopy.projections
    import matplotlib.pyplot as plt
    f = cartopy.projections.Mollweide()
    f = cartopy.projections.EquiRectangular()
    f = cartopy.projections.Hammer()
    f = cartopy.projections.Robinson()
    
    
    f = cartopy.projections.Sinusoidal()
    f = cartopy.projections.Cassini()
    f = cartopy.projections.Mercator()
    
    f = cartopy.projections.TransverseMercator()
    
    f = cartopy.projections.Polyconic()
    
    
    f = cartopy.projections.Miller()
    f = cartopy.projections.GallStereographic()
    
    f = cartopy.projections.LambertConformalConic()
    
    f = cartopy.projections.Stereographic()
    
#    f = cartopy.projections.EquidistantConic()
    
    f = cartopy.projections.AlbersEqualArea()
    
    f = cartopy.projections.VanderGrinten()
    
    f = cartopy.projections.McBrydeThomasFlatPolarQuartic()
    f = cartopy.projections.Collignon()
    f = cartopy.projections.PerspectiveConic()

    f = cartopy.projections.LambertConformalConic()
    f = cartopy.projections.LambertEqualArea()

    f = cartopy.projections.Bonne()
    f = cartopy.projections.Stereographic(lat_0=90)
    f = cartopy.projections.Ortelius()
    
    f = cartopy.projections.Lagrange()

    
#    f = cartopy.projections.Lambert()
#    f = cartopy.projections.Geos()

    # Half circle projections...
#    f = cartopy.projections.Ortho()
#    f = cartopy.projections.NearSided()
    
    
    plt.subplot(111, projection=f)
#    ax = plt.axes(projection=f)
    print plt.gca()
    plt.plot(range(10))
    plt.grid(True)
    plt.show()

#    print dir(f.proj)
#    print f.proj._getmapboundary()
#    print f.ymin