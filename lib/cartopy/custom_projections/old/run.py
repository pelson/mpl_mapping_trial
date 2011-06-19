import custom_projection
import matplotlib.pyplot as plt
import numpy as np

plt.subplot(111, projection="mollweide2")

p = plt.plot([-1, 1, 1, 2*np.pi - 0.5, 2*np.pi-1 ], [1, -1, 1, -1.3, 1], "o-")
#
##p = plt.plot([1, 2*np.pi+1 ], [1, -1.4], "o-")
#
plt.grid()

import matplotlib
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import matplotlib.path as mpath

# make up some data on a regular lat/lon grid.
lons = np.linspace(0, 2 * np.pi, 145)
lats = np.linspace(-np.pi/2, np.pi/2, 74)
lons, lats = np.meshgrid(lons, lats)

wave = 0.75*(np.sin(2.*lats)**8*np.cos(4.*lons))
mean = 0.5*np.cos(2.*lats)*((np.sin(2.*lats))**2 + 2.)

print 'doing contour'

#CS = plt.contour(lons, lats, wave+mean, 15, linewidths=1.5)
CS = plt.contour(lons, lats, wave+mean, 15)

######################################################

plt.show()