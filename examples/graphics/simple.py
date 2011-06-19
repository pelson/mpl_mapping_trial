import cartopy.projections
import matplotlib.pyplot as plt


def plot_lines():
    p = plt.plot([0, 360], [-26, 32], "o-")
    p = plt.plot([180, 180], [-89, 43], "o-")
    p = plt.plot([30, 210], [0, 0], "o-")
    plt.grid()
    

for proj in [cartopy.projections.Mollweide(), cartopy.projections.Hammer(), cartopy.projections.EquiRectangular()]:
    ax = plt.subplot(111, projection=proj)
    plot_lines()
    plt.show()    