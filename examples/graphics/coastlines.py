import cartopy.projections
import matplotlib.pyplot as plt


mol = cartopy.projections.Mollweide()
ax = plt.subplot(111, projection=mol)
ax.coastlines()
plt.show()