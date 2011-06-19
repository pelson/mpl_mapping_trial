"""
generate the rst files for the examples by iterating over the pylab examples
"""
import os, glob

import os
import re
import shutil
import sys
fileList = []

def out_of_date(original, derived):
    """
    Returns True if derivative is out-of-date wrt original,
    both of which are full file paths.

    TODO: this check isn't adequate in some cases.  Eg, if we discover
    a bug when building the examples, the original and derived will be
    unchanged but we still want to force a rebuild.
    """
    return (not os.path.exists(derived) or
            os.stat(derived).st_mtime < os.stat(original).st_mtime)

noplot_regex = re.compile(r"#\s*-\*-\s*noplot\s*-\*-")

def generate_example_rst(app):
    rootdir = os.path.join(app.builder.srcdir, '../../examples')
    exampledir = os.path.join(app.builder.srcdir, 'examples')
    shutil.rmtree(exampledir)
    shutil.copytree(rootdir, exampledir)
    
    rootdir = exampledir
    
    if not os.path.exists(exampledir):
        os.makedirs(exampledir)

    datad = {}
    print 'rootdir:', rootdir
    for root, subFolders, files in os.walk(rootdir):
        for fname in files:
            if ( fname.startswith('.') or fname.startswith('#')
                 or fname.startswith('_') or not fname.endswith('.py') ):
                continue

            fullpath = os.path.join(root,fname)
            contents = file(fullpath).read()
            # indent
            relpath = os.path.split(root)[-1]
            datad.setdefault(relpath, []).append((fullpath, fname, contents))

    subdirs = datad.keys()
    subdirs.sort()

    fhindex = file(os.path.join(exampledir, 'index.rst'), 'w')
    fhindex.write("""\
.. _examples-index:

####################
Examples
####################

.. toctree::
    :maxdepth: 2

""")

    for subdir in subdirs:
        rstdir = os.path.join(exampledir, subdir)
        if not os.path.exists(rstdir):
            os.makedirs(rstdir)

        outputdir = os.path.join(app.builder.outdir, 'examples')
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        outputdir = os.path.join(outputdir, subdir)
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        subdirIndexFile = os.path.join(rstdir, 'index.rst')
        fhsubdirIndex = file(subdirIndexFile, 'w')
        fhindex.write('    %s/index.rst\n\n'%subdir)

        fhsubdirIndex.write("""\
.. _%s-examples-index:

##############################################
%s Examples
##############################################

.. toctree::
    :maxdepth: 1

"""%(subdir, subdir))

        sys.stdout.write(subdir + ", ")
        sys.stdout.flush()

        data = datad[subdir]
        data.sort()

        for fullpath, fname, contents in data:
            basename, ext = os.path.splitext(fname)
            outputfile = os.path.join(outputdir, fname)
            #thumbfile = os.path.join(thumb_dir, '%s.png'%basename)
            #print '    static_dir=%s, basename=%s, fullpath=%s, fname=%s, thumb_dir=%s, thumbfile=%s'%(static_dir, basename, fullpath, fname, thumb_dir, thumbfile)

            rstfile = '%s.rst'%basename
            outrstfile = os.path.join(rstdir, rstfile)

            fhsubdirIndex.write('    %s\n'%rstfile)

            if not out_of_date(fullpath, outrstfile):
                continue

            fh = file(outrstfile, 'w')
            fh.write('.. _%s-%s:\n\n'%(subdir, basename))
            title = '%s example code: %s'%(subdir, fname)
            #title = '<img src=%s> %s example code: %s'%(thumbfile, subdir, fname)


            fh.write(title + '\n')
            fh.write('='*len(title) + '\n\n')

            do_plot = (subdir in ('graphics',
                                  ) and
                       not noplot_regex.search(contents))

            # indent the contents
            contents = '\n'.join(['    %s'%row.rstrip() for row in contents.split('\n')])

            if do_plot:
                fh.write("\n\n.. plot:: %s\n\n::\n\n" % fullpath)
#                fh.write("\n\n.. plot::\n\n")
#                fh.write(contents)
#                fh.write('\n\n..\n::\n\n')
            
            else:
                fh.write("[`source code <%s>`_]\n\n::\n\n" % fname)
                fhstatic = file(outputfile, 'w')
                fhstatic.write(contents)
                fhstatic.close()

            fh.write(contents)

            fh.close()

        fhsubdirIndex.close()

    fhindex.close()


def setup(app):
    app.connect('builder-inited', generate_example_rst)
