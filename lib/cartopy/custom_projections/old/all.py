import custom_projection
import matplotlib.pyplot as plt
import numpy as np


plt.subplot(111, projection="mollweide2")

p = plt.plot([-1, 1, 1, 2*np.pi +1 ], [-1, -1, 1, -0.5], "o-")
p = plt.plot([-1, 1, 1, 2*np.pi - 0.5, 2*np.pi-1 ], [1, -1, 1, -1.3, 1], "o-")

plt.grid(True)


import matplotlib
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

plt.subplot(111, projection="mollweide2")
collection = PatchCollection([mpatches.RegularPolygon( (3, 0), 5, 1.8)], cmap=matplotlib.cm.jet, alpha=0.4)
plt.gca().add_collection(collection)
#plt.show()



import matplotlib
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import matplotlib.path as mpath

plt.subplot(111, projection="mollweide2")
# add a path patch
Path = mpath.Path
verts = np.array([
     (0, 0),
     (0.8 * 2, 0.8),
     (4, 0),
     (0.8 * 2, -0.8),
     (0, 0),
    ]) * 1.5
codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO]

path = mpath.Path(verts, codes)

collection = PatchCollection([mpatches.PathPatch(path)], cmap=matplotlib.cm.jet, alpha=0.4)
plt.gca().add_collection(collection)
plt.grid()




##########################################################


# make up some data on a regular lat/lon grid.
lons = np.linspace(0, 2 * np.pi, 145)
lats = np.linspace(-np.pi/2, np.pi/2, 74)
lons, lats = np.meshgrid(lons, lats)

wave = 0.75*(np.sin(2.*lats)**8*np.cos(4.*lons))
mean = 0.5*np.cos(2.*lats)*((np.sin(2.*lats))**2 + 2.)

print 'doing contour'

#CS = plt.contour(lons, lats, wave+mean, 15, linewidths=1.5)
CS = plt.contourf(lons, lats, wave+mean, 15)


##########################################################


import matplotlib
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import matplotlib.path as mpath

plt.subplot(111, projection="mollweide2")

# make up some data on a regular lat/lon grid.
lons = np.linspace(0, 2 * np.pi, 145)
lats = np.linspace(-np.pi/2, np.pi/2, 74)
lons, lats = np.meshgrid(lons, lats)

wave = 0.75*(np.sin(2.*lats)**8*np.cos(4.*lons))
mean = 0.5*np.cos(2.*lats)*((np.sin(2.*lats))**2 + 2.)


CS = plt.quiver(lons, lats, lons, lons*0)

#############################################



plt.show()