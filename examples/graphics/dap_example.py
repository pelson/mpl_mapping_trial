import cartopy.projections
import matplotlib.pyplot as plt
import numpy as np

from pydap.client import open_url


dataset = open_url('http://test.opendap.org/dap/data/nc/coads_climatology.nc')
var = dataset['SST']

x = var.COADSX[:]
y = var.COADSY[:]
data = var.SST[0, :, :]

data[data == -1e+34] = 0#np.NaN

# Append the first line to the end of the data
data = np.concatenate([data, data[:, 0:1]], 1)
x = np.append(x, x[0] + 360)

mol = cartopy.projections.Mollweide()
ax = plt.subplot(121, projection=mol)
plt.contourf(x, y, data, 50)
ax.coastlines()
plt.show()

#ax = plt.subplot(112, projection=mol)
#plt.pcolor(x, y, data)
#ax.coastlines()
plt.show()