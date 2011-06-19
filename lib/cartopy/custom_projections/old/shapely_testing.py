from shapely.geometry import polygon, linestring

p = polygon.Polygon(((0, 0), (0, 1), (1, 1)))

p2 = polygon.Polygon(((0, 0), (0, 0.5), (0.5, 0.5)))

line_r = linestring.LineString(((-0.5, 0.5), (0.5, -0.25)))

print p


print p.difference(p2)


print dir(p)

print p.difference(p2).exterior.xy

import matplotlib.pyplot as plt

#plt.plot(*p.exterior.xy, color='red')
#plt.plot(*p2.exterior.xy, color='blue')
plt.plot(*p.intersection(p2).exterior.xy, color='yellow')

line_r.intersection(p2)

#plt.plot(*line_r.xy)

plt.plot(*line_r.intersection(p2).xy)

plt.show()
