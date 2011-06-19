
Example Mercator
=====================================================================================
            
.. plot:: /home/phil/Development/Python/cartopy/lib/cartopy/../../docs/source/examples/graphics/Mercator_simple_lines.py

::
    import matplotlib.pyplot as plt
    import cartopy.projections as prj
    
    
    proj = prj.Mercator()
    plt.axes(projection=proj)
    p = plt.plot([0, 360], [-26, 32], "o-")
    p = plt.plot([180, 180], [-89, 43], "o-")
    p = plt.plot([30, 210], [0, 0], "o-")
    plt.title('Mercator')
    plt.grid()
    plt.show()
    
            