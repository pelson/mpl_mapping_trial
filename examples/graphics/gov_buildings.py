import cartopy.projections
import matplotlib.pyplot as plt
from geopy import geocoders


mol = cartopy.projections.Mollweide()
ax = plt.subplot(111, projection=mol)

g = geocoders.Google()

addresses = ['10 Downing Street, London',
             '1600 Pennsylvania Avenue, Washington D.C.',
            ]

for address in addresses:
    place, (lat, lng) = g.geocode(address)
    bbox_args = dict(boxstyle="round", fc="0.8")
    align = 'left' if address == '10 Downing Street, London' else 'right'
    plt.gca().annotate('\n'.join(place.split(',')), 
                       xy=(lng, lat),
                       ha=align,
                       bbox=bbox_args,
                       )

ax.coastlines()
plt.show()