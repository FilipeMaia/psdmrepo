#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: template!SConscript! 8 2008-10-08 22:35:33Z salnikov $
#
# Description:
#  SConscript file for package hdf5
#------------------------------------------------------------------------

# Do not delete following line, it must be present in 
# SConscript file for any SIT project
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
PREFIX = pjoin('$SIT_EXTERNAL_SW', "szip", szip_ver)
LIBDIR = "lib"
PKGLIBS = "sz"
standardExternalPackage('szip', **locals())


hdf5_ver = "1.8.14"
PREFIX = pjoin('$SIT_EXTERNAL_SW', "hdf5", hdf5_ver)
INCDIR = "include"
LIBDIR = "lib"
BINDIR = "bin"
PKGLIBS = "hdf5 hdf5_cpp hdf5_hl hdf5_hl_cpp"
DEPS = "szip"
standardExternalPackage('hdf5', **locals())

h5check_ver = "2.0.1"
PREFIX = pjoin('$SIT_EXTERNAL_SW', "h5check", h5check_ver)
INCDIR = None
LIBDIR = None
BINDIR = "bin"
standardExternalPackage('h5check', **locals())

