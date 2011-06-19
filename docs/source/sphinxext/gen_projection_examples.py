"""
generate the rst files for the examples by iterating over the pylab examples
"""
import os, glob

import os
import re
import shutil
import sys

import cartopy
import cartopy.cbook


scenarios = {}


scenarios['simple_lines'] = """import matplotlib.pyplot as plt
import cartopy.projections as prj


proj = <@@proj_code@@>
plt.axes(projection=proj)
p = plt.plot([0, 360], [-26, 32], "o-")
p = plt.plot([180, 180], [-89, 43], "o-")
p = plt.plot([30, 210], [0, 0], "o-")
plt.title('<@@proj_name@@>')
plt.grid()
plt.show()
"""



scenarios['polygon'] = """import matplotlib.pyplot as plt
import cartopy.projections as prj


from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import matplotlib.cm
from matplotlib.path import Path


proj = <@@proj_code@@>
plt.axes(projection=proj)

poly = mpatches.RegularPolygon( (140, 10), 4, 81.0)
pth = Path([[0, 45], [300, 45], [300, -45], [0, -45], [0, -45], [200, 20], [150, 20], [150, -20], [200, -20], [200, -20]], [1, 2, 2, 2, 79, 1, 2, 2 ,2, 79])
poly = mpatches.PathPatch(pth)
collection = PatchCollection([poly], cmap=matplotlib.cm.jet, alpha=0.4)
plt.gca().add_collection(collection)

plt.grid()

plt.gca().coastlines()
plt.title('<@@proj_name@@>')
plt.show()
"""

scenarios['contourf'] = '''import matplotlib.pyplot as plt
import cartopy.projections as prj


proj = <@@proj_code@@>
ax = plt.subplot(111, projection=proj)

# make up some data on a regular lat/lon grid.
lons = np.linspace(0, 360, 145)
lats = np.linspace(-90, 90, 74)
lons, lats = np.meshgrid(lons, lats)

lons_rad = np.linspace(0, 2 * np.pi, 145)
lats_rad = np.linspace(-np.pi, np.pi, 74)
lons_rad, lats_rad = np.meshgrid(lons_rad, lats_rad)

wave = 0.75*(np.sin(2.*lats_rad)**8*np.cos(4.*lons_rad))
mean = 0.5*np.cos(2.*lats_rad)*((np.sin(2.*lats_rad))**2 + 2.)

plt.contourf(lons, lats, wave+mean, 10, alpha=0.9)

plt.grid()
plt.title('<@@proj_name@@>')
plt.show()
'''


def generate_projection_examples(app):
#    exampledir = os.path.join(app.builder.srcdir, 'examples', 'graphics')
    exampledir = os.path.join(os.path.dirname(cartopy.__file__), '..', '..', 'docs', 'source', 'examples', 'graphics')
    
    shutil.rmtree(exampledir)
    
    if not os.path.exists(exampledir):
        os.makedirs(exampledir)

    fpaths = []

    datad = {}
    for proj in cartopy.cbook.projections:
        
        proj_name = proj.__class__.__name__
        proj_code = proj._repr_full_import()
        proj_info = {'proj_name': proj_name,
                     'proj_code': proj_code.replace('cartopy.projections', 'prj')}
        print proj_code
        for scenario_name, scenario_code in scenarios.iteritems():
#            print 'doing %s with proj %s' % (scenario_name, proj_name)
            code = scenario_code
            for replacement_name, replace_with in proj_info.iteritems():
                print replacement_name, replace_with
                code = code.replace('<@@%s@@>' % replacement_name, replace_with)
            code_fname = proj_name + '_' + scenario_name + '.py'
            code_fpath = os.path.join(exampledir, code_fname)
            fh = open(code_fpath, 'w')
            fh.write(code)
            fh.close()
            
            fname = proj_name + '_' + scenario_name + '.rst'
            fpath = os.path.join(exampledir, fname)
            
            fh = open(fpath, 'w')
            fpaths.append(fpath)
            
            rst = """
Example %s
=====================================================================================
            
.. plot:: %s

::
    %s
            """ % (proj_name, code_fpath, code.replace('\n', '\n    '))
            fh.write(rst)
            fh.close()
    
    
    fpath = os.path.join(exampledir, 'index.rst')
    index = """
####################
Examples
####################

.. toctree::
    :maxdepth: 2

    """ 
    index += '\n    '.join([os.path.basename(p) for p in fpaths])
   
    fh = open(fpath, 'w')
    fh.write(index)
    fh.close()
    
            
#                
#        fh = file(outrstfile, 'w')
#            fh.write('.. _%s-%s:\n\n'%(subdir, basename))
#            title = '%s example code: %s'%(subdir, fname)
#            #title = '<img src=%s> %s example code: %s'%(thumbfile, subdir, fname)
#
#
#            fh.write(title + '\n')
#            fh.write('='*len(title) + '\n\n')
#
#            do_plot = (subdir in ('graphics',
#                                  ) and
#                       not noplot_regex.search(contents))
#
#            # indent the contents
#            contents = '\n'.join(['    %s'%row.rstrip() for row in contents.split('\n')])
#
#            if do_plot:
#                fh.write("\n\n.. plot:: %s\n\n::\n\n" % fullpath)
##                fh.write("\n\n.. plot::\n\n")
##                fh.write(contents)
##                fh.write('\n\n..\n::\n\n')
#            
#            else:
#                fh.write("[`source code <%s>`_]\n\n::\n\n" % fname)
#                fhstatic = file(outputfile, 'w')
#                fhstatic.write(contents)
#                fhstatic.close()
#
#            fh.write(contents)
#
#            fh.close()
#
#        fhsubdirIndex.close()
#
#    fhindex.close()


def setup(app):
    app.connect('builder-inited', generate_projection_examples)


if __name__ == '__main__':
    generate_projection_examples(None)