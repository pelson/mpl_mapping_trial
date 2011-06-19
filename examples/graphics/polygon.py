import cartopy.projections
import matplotlib.pyplot as plt

from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import matplotlib.cm
from matplotlib.path import Path



def add_holey_polygon():
    pth = Path([[0, 115], [300, 45], [300, -45], [0, -45], [0, -45], [200, 20], [150, 20], [150, -20], [200, -20], [200, -20]], [1, 2, 2, 2, 79, 1, 2, 2 ,2, 79])
    poly = mpatches.PathPatch(pth)
    collection = PatchCollection([poly], cmap=matplotlib.cm.jet, alpha=0.4)
    plt.gca().add_collection(collection)
    
    plt.grid()


proj = cartopy.projections.Polyconic()
proj = cartopy.projections.EquiRectangular()

ax = plt.subplot(111, projection=proj)
add_holey_polygon()
#ax.coastlines()
plt.show()