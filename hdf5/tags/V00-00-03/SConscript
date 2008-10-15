#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: template!SConscript! 8 2008-10-08 22:35:33Z salnikov $
#
# Description:
#  SConscript file for package hdf5
#------------------------------------------------------------------------

# Do not delete following line, it must be present in 
# SConscript file for any LUSI project
Import('*')

import os
from os.path import join as pjoin
from SConsTools.standardExternalPackage import standardExternalPackage

#
# For the standard external packages which contain includes, libraries, 
# and applications it is usually sufficient to call standardExternalPackage()
# giving some or all parameters.
#

szip_ver = "2.1"
PREFIX  = pjoin(env['LUSI_ROOT'],"sw/external/szip",szip_ver,env['LUSI_ARCH'])
LIBDIR  = "lib"
PKGLIBS = "sz" 
standardExternalPackage ( 'szip', **locals() )


mxml_ver = "2.5"
PREFIX  = pjoin(env['LUSI_ROOT'],"sw/external/mxml",mxml_ver,env['LUSI_ARCH'])
LIBDIR  = "lib"
PKGLIBS = "mxml m"
standardExternalPackage ( 'mxml', **locals() )


hdf5_ver = "1.8.1"
PREFIX  = pjoin(env['LUSI_ROOT'],"sw/external/hdf5",hdf5_ver,env['LUSI_ARCH'])
INCDIR  = "include"
LIBDIR  = "lib"
BINDIR  = "bin"
PKGLIBS = "hdf5"
DEPS    = "szip mxml"
standardExternalPackage ( 'hdf5', **locals() )

