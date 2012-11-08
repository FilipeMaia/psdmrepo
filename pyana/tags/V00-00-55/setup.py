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
    name         = 'pyana',
    version      = _VERSION_STRING,
    author       = "Andy Salnikov",
    author_email = "salnikov@slac.stanford.edu",
    description  = "Python analysis package for LCLS",
    )


#sources = glob.glob( os.path.join(SRC_DIR,"*.cpp") ) + \
#    glob.glob( os.path.join(SRC_DIR,"types","*","*.cpp") )
#
#module = Extension('_pyana',
#                   sources = sources,
#                   include_dirs = ['src']
#                   )
#
#
#options['ext_modules'] = [module]

options['packages'] = ['pyana']
options['scripts'] = [ 'scripts/pyana' ]

setup(**options)

