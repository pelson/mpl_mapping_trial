import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from matplotlib.collections import PatchCollection
import matplotlib.cm
import numpy
import os


def tissot(lon_0, lat_0, radius_deg, npts)
        ax = kwargs.pop('ax', None) or self._check_ax()
        g = pyproj.Geod(a=self.rmajor,b=self.rminor)
        az12,az21,dist = g.inv(lon_0,lat_0,lon_0,lat_0+radius_deg)
        seg = [self(lon_0,lat_0+radius_deg)]
        delaz = 360./npts
        az = az12
        for n in range(npts):
            az = az+delaz
            # skip segments along equator (Geod can't handle equatorial arcs)
            if np.allclose(0.,lat_0) and (np.allclose(90.,az) or np.allclose(270.,az)):
                continue
            else:
                lon, lat, az21 = g.fwd(lon_0, lat_0, az, dist)
            x,y = self(lon,lat)
            # add segment if it is in the map projection region.
            if x < 1.e20 and y < 1.e20:
                seg.append((x,y))
        poly = Polygon(seg,**kwargs)
        ax.add_patch(poly)
        # set axes limits to fit map region.
        self.set_axes_limits(ax=ax)
        return poly


if __name__ == '__main__':
    import matplotlib.patches
    import cartopy.projections as prj
    f = prj.Mollweide()
    
    plt.subplot(111, projection=f)
    
    add_coastlines()
        
#    plt.grid()
    plt.show()  