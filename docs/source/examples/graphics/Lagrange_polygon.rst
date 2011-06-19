
Example Lagrange
=====================================================================================
            
.. plot:: /home/phil/Development/Python/cartopy/lib/cartopy/../../docs/source/examples/graphics/Lagrange_polygon.py

::
    import matplotlib.pyplot as plt
    import cartopy.projections as prj
    
    
    from matplotlib.collections import PatchCollection
    import matplotlib.patches as mpatches
    import matplotlib.cm
    from matplotlib.path import Path
    
    
    proj = prj.Lagrange()
    plt.axes(projection=proj)
    
    poly = mpatches.RegularPolygon( (140, 10), 4, 81.0)
    pth = Path([[0, 45], [300, 45], [300, -45], [0, -45], [0, -45], [200, 20], [150, 20], [150, -20], [200, -20], [200, -20]], [1, 2, 2, 2, 79, 1, 2, 2 ,2, 79])
    poly = mpatches.PathPatch(pth)
    collection = PatchCollection([poly], cmap=matplotlib.cm.jet, alpha=0.4)
    plt.gca().add_collection(collection)
    
    plt.grid()
    
    plt.gca().coastlines()
    plt.title('Lagrange')
    plt.show()
    
            