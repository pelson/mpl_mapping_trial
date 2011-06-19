if __name__ == '__main__':
    import custom_projection
    import matplotlib.pyplot as plt
    import numpy as np
    
#    plt.subplot(111, projection="lambert2")
    plt.subplot(111, projection="mollweide2")
#    p = plt.plot([-1, 1, 1, 2*np.pi - 0.5, -1 ], [1, -1, 1, -1.3, 1], "o-")
    p = plt.plot(np.rad2deg([0, 8*np.pi, ]), np.rad2deg([-np.pi/4, np.pi/4]), "o-")
    
    
    import matplotlib
    from matplotlib.collections import PatchCollection
    import matplotlib.patches as mpatches
    
##    plt.subplot(111, projection="mollweide2")
#    collection = PatchCollection([mpatches.RegularPolygon( (np.pi, 0), 5, 1.0)], cmap=matplotlib.cm.jet, alpha=0.4)
#    plt.gca().add_collection(collection)
    
    # make up some data on a regular lat/lon grid.
    lons = np.linspace(0, 2 * np.pi, 145)
    lats = np.linspace(-np.pi/2, np.pi/2, 74)
    lons, lats = np.meshgrid(lons, lats)
    
    wave = 0.75*(np.sin(2.*lats)**8*np.cos(4.*lons))
    mean = 0.5*np.cos(2.*lats)*((np.sin(2.*lats))**2 + 2.)
    
    print 'doing contour'
    
    #CS = plt.contour(lons, lats, wave+mean, 15, linewidths=1.5)
#    CS = plt.contourf(lons, lats, wave+mean, 15, alpha=0.5)
    
    plt.grid()
    
    plt.show()