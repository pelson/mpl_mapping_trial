import matplotlib.pyplot as plt
import cartopy.projections as prj


proj = prj.Bonne()
plt.axes(projection=proj)
p = plt.plot([0, 360], [-26, 32], "o-")
p = plt.plot([180, 180], [-89, 43], "o-")
p = plt.plot([30, 210], [0, 0], "o-")
plt.title('Bonne')
plt.grid()
plt.show()
