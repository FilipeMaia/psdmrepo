#
# $Id$
#
# Copyright (c) 2010-2008 SLAC National Accelerator Laboratory
# 

from distutils.core import setup, Extension
from distutils.command.build import build
import os
import glob

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


module = Extension('pdsdata',
                   sources = glob.glob( os.path.join(SRC_DIR,"*.cpp") )
                   )

# we need a special
class PdsdataBuilder( build ):
    
    build.user_options.append( ('pdsdata-dir=', None, "pdsdata top level installation directory") )
    build.user_options.append( ('pdsdata-incdir=', None, "pdsdata include directory relative to pdsdata-dir") )
    build.user_options.append( ('pdsdata-libdir=', None, "pdsdata libraries directory relative to pdsdata-dir") )
    build.user_options.append( ('pdsdata-libs=', None, "list pdsdata libraries") )
    
    def initialize_options (self):
        build.initialize_options(self)
        self.pdsdata_dir = None
        self.pdsdata_incdir = None
        self.pdsdata_libdir = None
        self.pdsdata_libs = None

    def finalize_options (self):
        build.finalize_options(self)
        
        #adjust the paths and libs
        if self.pdsdata_dir :
            
            global _incdirs
            module.include_dirs.append( os.path.join(self.pdsdata_dir, self.pdsdata_incdir) )

            global _libdirs
            module.library_dirs.append( os.path.join(self.pdsdata_dir, self.pdsdata_libdir) )

            global _libs
            module.libraries.extend( self.pdsdata_libs.split() )


options['ext_modules'] = [module]
options['cmdclass'] = dict( build = PdsdataBuilder )

setup(**options)

