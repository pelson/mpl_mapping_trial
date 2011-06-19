
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.cbook
from matplotlib.projections.polar import PolarAxes



def path_interpolation(path, n_steps):
            path_verts, path_codes =  zip(*list(path.iter_segments(curves=False)))
            path_verts = np.array(path_verts)
            path_codes = np.array(path_codes)
            verts_split_inds = np.where(path_codes == Path.MOVETO)[0]
            verts_split = np.split(path_verts, verts_split_inds, 0)
            codes_split = np.split(path_codes, verts_split_inds, 0)
            
            v_collection = []
            c_collection = []
            for path_verts, path_codes in zip(verts_split, codes_split):
                if len(path_verts) == 0:
                    continue
                
#                print path_verts.shape
                verts = matplotlib.cbook.simple_linear_interpolation(path_verts, n_steps)
                v_collection.append(verts)
#                print verts.shape
                codes = np.ones(verts.shape[0]) * Path.LINETO
                codes[0] = Path.MOVETO
                c_collection.append(codes)
                
            return Path(np.concatenate(v_collection), np.concatenate(c_collection))
        

pth = Path(np.array([(0.158, -0.257), (0.035, -0.11)]))

n = 8
#theta = np.linspace(-np.pi, np.pi, n+1)[:-1]
#x = np.sin(theta)
#y = np.cos(theta)
y = np.linspace(0, 1, 3)
x = np.linspace(0, 6, 3)

pth = Path(zip(x, y))

pth = path_interpolation(pth, 2000)

pth = PolarAxes.PolarTransform().transform_path(pth)
plt.plot(np.arange(3)-1, np.arange(3)-1)
plt.gca().add_patch(mpatches.PathPatch(pth, fill=False))
plt.show()
