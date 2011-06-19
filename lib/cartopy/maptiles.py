import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
from globalmaptiles import GlobalMercator


mercator = GlobalMercator()


TILE_LAT_RANGE = (-85.0511287798065, 85.0511287798065)
TILE_LON_RANGE = (-179.999999999999, 180)

MIN_ZOOM = 0
MAX_ZOOM = 7

current_maptile_system = None

current_map=None


def setup_region(lon_range=TILE_LON_RANGE, lat_range=TILE_LAT_RANGE, min_zoom=MIN_ZOOM, max_zoom=MAX_ZOOM):
    
    uv_lower = mercator.LatLonToMeters(lat_range[0], lon_range[0])
    uv_upper = mercator.LatLonToMeters(lat_range[1], lon_range[1])    
    
    min_x, max_x, min_y, max_y = google_tiles_in_range(uv_lower, uv_upper, min_zoom)

    map = setup_googletile_region(x_tile_range=[min_x, max_x], y_tile_range=[min_y, max_y], 
                      min_zoom=min_zoom, max_zoom=max_zoom)
    
    global current_maptile_system
    current_maptile_system = [map, lon_range, lat_range, min_zoom, max_zoom]
     
     
def setup_googletile_region(x_tile_range=(0, 0), y_tile_range=(0,0), min_zoom=0, max_zoom=7):
    # define ranges based on min_zoom zoom level
    
    lower_bounds_meters = mercator.TileBounds(x_tile_range[0], y_tile_range[0], min_zoom)[:2]
    upper_bounds_meters = mercator.TileBounds(x_tile_range[1], y_tile_range[1], min_zoom)[2:]
    lat_min, lon_min = mercator.MetersToLatLon(*lower_bounds_meters)
    lat_max, lon_max = mercator.MetersToLatLon(*upper_bounds_meters)
    
    f = plt.figure(figsize=(1, 1), dpi=512)
    margins = [0, 0, 1, 1]
 
    # Setup axes
    axes = plt.Axes(f, margins, frameon = False)
    
    # Add axes to figure
    f.add_axes(axes) 
    return Basemap(projection='merc', 
                   llcrnrlon=lon_min,
                   llcrnrlat=lat_min,
                   urcrnrlon=lon_max,
                   urcrnrlat=lat_max,
#                   resolution='f', 
                   )
#    return iplt.map_setup(lon_range=[lon_min, lon_max], lat_range=[lat_min, lat_max], projection='merc',
#                          resolution='f', 
#                          )    
   

def google_tiles_in_range(meters_lower, meters_upper, zoom):
    
    x_min, y_min = mercator.GoogleTile(*mercator.MetersToTile(meters_lower[0], meters_lower[1], zoom) + (zoom,))
    x_max, y_max = mercator.GoogleTile(*mercator.MetersToTile(meters_upper[0], meters_upper[1], zoom) + (zoom,))
        
    return x_min, x_max, y_min, y_max 


def google_tiles_in_range_generator(lon_range, lat_range, min_zoom=0, max_zoom=2):
    """
    Return all tiles in the bounding box provided between min_zoom and max_zoom.
    """
    
    uv_lower = mercator.LatLonToMeters(lat_range[0], lon_range[0])
    uv_upper = mercator.LatLonToMeters(lat_range[1], lon_range[1])
    
    for zoom in range(min_zoom, max_zoom+1):
        x_min, x_max, y_min, y_max = google_tiles_in_range(uv_lower, uv_upper, zoom)
                
        # swap if y min is smaller than y max (this could be the case because of the google maps switch)
        if y_min > y_max:
            y_min, y_max = (y_max, y_min)
        
        for x in range(x_min, x_max+1):
            for y in range(y_min, y_max+1):
                yield (x, y, zoom)

def save_tiles(filename_template, print_current_tile=False):
    """
    Sample filename_template:
        
        '/home/h02/itpe/public_html/sample_gmt/name2/z%(z)s_y%(y)s_x%(x)s.png'
    """
    
    for x, y, z in google_tiles_in_range_generator(*current_maptile_system[1:]):
        
        fname = filename_template % {'x':x, 'y':y, 'z':z}
            
        if print_current_tile:
            print 'Doing tile z:%s, y:%s, x:%s to %s' % (z, y, x, fname)
        
        tx, ty = mercator.GoogleTile(x, y, z)
        t_bounds = mercator.TileLatLonBounds( tx, ty, z)
        
        lon_range = (t_bounds[1], t_bounds[3])
        lat_range = (t_bounds[0], t_bounds[2])
        
        u_meters, v_meters = current_maptile_system[0](lon_range, lat_range)

        plt.gca().set_xlim(*u_meters)
        plt.gca().set_ylim(*v_meters)
        
        # Remove the previous display
        # This should be done automatically by MPL, but it is not... 
        # (http://old.nabble.com/Transparency-with-fig.canvas.mpl_connect-td27724532.html)
        plt.gcf().canvas.get_renderer().clear()
         
        plt.savefig(fname, dpi=512, transparent=True)
        

if __name__ == '__main__':
  
#    setup_region(lon_range=[0, 73], lat_range=[20, 78], min_zoom=0, max_zoom=4)
    setup_region(lon_range=[0, 360], lat_range=[-80, 80], min_zoom=0, max_zoom=7)

    import numpy as np
    
    # make up some data on a regular lat/lon grid.
    nlats = 73; nlons = 145; delta = 2.*np.pi/(nlons-1)
    lats = (0.5*np.pi-delta*np.indices((nlats,nlons))[0,:,:])
    lons = (delta*np.indices((nlats,nlons))[1,:,:])
    wave = 0.75*(np.sin(2.*lats)**8*np.cos(4.*lons))
    mean = 0.5*np.cos(2.*lats)*((np.sin(2.*lats))**2 + 2.)
    # compute native map projection coordinates of lat/lon grid.
    x, y = current_maptile_system[0](lons*180./np.pi, lats*180./np.pi)
    # contour data over the map.
    CS = current_maptile_system[0].contour(x,y,wave+mean,15,linewidths=1.5)

    current_maptile_system[0].drawcoastlines()

    save_tiles('/home/phil/Development/cartopy/sample_gmt/rotated/z%(z)s_y%(y)s_x%(x)s.png', print_current_tile=True)