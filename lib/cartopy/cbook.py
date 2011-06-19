import cartopy.projections # as prj

#projections = [prj.EquiRectangular(), prj.Mollweide(), prj.Hammer()]

projections = [
                cartopy.projections.Mollweide(),
                cartopy.projections.EquiRectangular(),
                cartopy.projections.Hammer(),
                cartopy.projections.Robinson(),
                cartopy.projections.Sinusoidal(),
                cartopy.projections.Cassini(),
                cartopy.projections.Mercator(),
                cartopy.projections.TransverseMercator(),
                cartopy.projections.Polyconic(),
                cartopy.projections.Miller(),
                cartopy.projections.GallStereographic(),
                cartopy.projections.LambertConformalConic(),    
                cartopy.projections.Stereographic(),
                cartopy.projections.AlbersEqualArea(),
                cartopy.projections.VanderGrinten(),
                cartopy.projections.McBrydeThomasFlatPolarQuartic(),
                cartopy.projections.LambertAzimuthal(),
                cartopy.projections.Collignon(),
                cartopy.projections.Bonne(),
                cartopy.projections.LambertEqualArea(),
                cartopy.projections.Ortelius(),
                cartopy.projections.Lagrange(),
                ]  