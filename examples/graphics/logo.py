"""
Thanks to Tony Yu <tsyu80@gmail.com> for the logo design
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from pylab import rand

mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.edgecolor'] = 'gray'


axalpha = 0.05
#figcolor = '#EFEFEF'
figcolor = 'white'
dpi = 80
fig = plt.figure(figsize=(6, 1.1),dpi=dpi)
fig.figurePatch.set_edgecolor(figcolor)
fig.figurePatch.set_facecolor(figcolor)


def add_math_background():
    ax = fig.add_axes([0., 0., 1., 1.])

    text = []
    text.append((r"$W^{3\beta}_{\delta_1 \rho_1 \sigma_2} = U^{3\beta}_{\delta_1 \rho_1} + \frac{1}{8 \pi 2} \int^{\alpha_2}_{\alpha_2} d \alpha^\prime_2 \left[\frac{ U^{2\beta}_{\delta_1 \rho_1} - \alpha^\prime_2U^{1\beta}_{\rho_1 \sigma_2} }{U^{0\beta}_{\rho_1 \sigma_2}}\right]$", (0.7, 0.2), 20))
    text.append((r"$\frac{d\rho}{d t} + \rho \vec{v}\cdot\nabla\vec{v} = -\nabla p + \mu\nabla^2 \vec{v} + \rho \vec{g}$",
                (0.35, 0.9), 20))
    text.append((r"$\int_{-\infty}^\infty e^{-x^2}dx=\sqrt{\pi}$",
                (0.15, 0.3), 25))
    #text.append((r"$E = mc^2 = \sqrt{{m_0}^2c^4 + p^2c^2}$",
    #            (0.7, 0.42), 30))
    text.append((r"$F_G = G\frac{m_1m_2}{r^2}$",
                (0.85, 0.7), 30))
    for eq, (x, y), size in text:
        ax.text(x, y, eq, ha='center', va='center', color="#11557c", alpha=0.25,
                transform=ax.transAxes, fontsize=size)
    ax.set_axis_off()
    return ax

def add_matplotlib_text(ax):
    ax.text(0.95, 0.5, 'cartopy', color='#11557c', fontsize=65,
               ha='right', va='center', alpha=1.0)

def add_polar_bar():
    ax = fig.add_axes([0.025, 0.075, 0.2, 0.85], resolution=50)
    ax.axesPatch.set_alpha(axalpha)
    ax.set_axisbelow(True)

if __name__ == '__main__':
    # maybe include a sextant?
    
    main_axes = add_math_background()
    add_matplotlib_text(main_axes)
    plt.show()