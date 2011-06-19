import matplotlib.pyplot as plt
import cartopy.projections as prj


proj = prj.TransverseMercator()
ax = plt.subplot(111, projection=proj)

# make up some data on a regular lat/lon grid.
lons = np.linspace(0, 360, 145)
lats = np.linspace(-90, 90, 74)
lons, lats = np.meshgrid(lons, lats)

lons_rad = np.linspace(0, 2 * np.pi, 145)
lats_rad = np.linspace(-np.pi, np.pi, 74)
lons_rad, lats_rad = np.meshgrid(lons_rad, lats_rad)

wave = 0.75*(np.sin(2.*lats_rad)**8*np.cos(4.*lons_rad))
mean = 0.5*np.cos(2.*lats_rad)*((np.sin(2.*lats_rad))**2 + 2.)

plt.contourf(lons, lats, wave+mean, 10, alpha=0.9)

plt.grid()
plt.title('TransverseMercator')
plt.show()
