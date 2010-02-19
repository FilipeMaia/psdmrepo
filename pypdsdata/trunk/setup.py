#
# $Id$
#
# Copyright (c) 2010 SLAC National Accelerator Laboratory
# 

from distutils.core import setup, Extension
from distutils.command.build import build
from distutils.sysconfig import get_python_inc
import os
import glob
import sys

VERSION_MAJOR = 0
VERSION_MINOR = 1

SRC_DIR = "src"

_VERSION_STRING = "%d.%d" % (VERSION_MAJOR, VERSION_MINOR) 


options = dict(
    name         = 'pypdsdata',
    version      = _VERSION_STRING,
    author       = "Andy Salnikov",
    author_email = "salnikov@slac.stanford.edu",
    description  = "Python interface for pdsdata XTC library",
    )


sources = glob.glob( os.path.join(SRC_DIR,"*.cpp") ) + \
    glob.glob( os.path.join(SRC_DIR,"types","*","*.cpp") )

module = Extension('_pdsdata',
                   sources = sources,
                   include_dirs = ['src']
                   )

# we need a special
class PdsdataBuilder( build ):
    
    build.user_options.append( ('pdsdata-dir=', None, "pdsdata top level installation directory") )
    build.user_options.append( ('pdsdata-incdir=', None, "pdsdata include directory relative to pdsdata-dir") )
    build.user_options.append( ('pdsdata-libdir=', None, "pdsdata libraries directory relative to pdsdata-dir") )
    build.user_options.append( ('pdsdata-libs=', None, "list pdsdata libraries") )
    build.user_options.append( ('numpy-incdir=', None, "NumPy top level installation directory") )
    
    def initialize_options (self):
        build.initialize_options(self)
        self.pdsdata_dir = None
        self.pdsdata_incdir = None
        self.pdsdata_libdir = None
        self.pdsdata_libs = None
        self.numpy_incdir = None

    def finalize_options (self):
        build.finalize_options(self)
        
        #adjust the paths and libs
        if self.pdsdata_dir :
            
            module.include_dirs.append( os.path.join(self.pdsdata_dir, self.pdsdata_incdir) )
            module.library_dirs.append( os.path.join(self.pdsdata_dir, self.pdsdata_libdir) )
            module.libraries.extend( self.pdsdata_libs.split() )


        if self.numpy_incdir :
            
            module.include_dirs.append( self.numpy_incdir )

        else :
            
            # try to locate NumPy installation

            header = "numpy/arrayobject.h"
            numpyinc = None

            # check standard Python include directory
            dir = os.path.join(get_python_inc(plat_specific=1), 'numpy')
            if os.path.isfile(os.path.join(dir,header)) :
                
                numpyinc = dir
                
            else :
                
                for dir in sys.path :
                    
                    subdir = dir
                    if os.path.isfile(os.path.join(subdir,header)) :
                        numpyinc = subdir
                        break
                    
                    subdir = os.path.join(dir,'numpy','core','include')
                    if os.path.isfile(os.path.join(subdir,header)) :
                        numpyinc = subdir
                        break
                    
            if not numpyinc :
                
                raise RuntimeError("Failed to find NumPy header file "+header+
                              ", use numpy-dir option to specify location of NumPy installation")
                
            module.include_dirs.append( os.path.join(numpyinc) )
      

options['ext_modules'] = [module]
options['packages'] = ['pdsdata']
options['scripts'] = [ 'scripts/pyxtcreader' ]
options['cmdclass'] = dict( build = PdsdataBuilder )

setup(**options)

