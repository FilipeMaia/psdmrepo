#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  SConscript file for package numpy
#------------------------------------------------------------------------

# Do not delete following line, it must be present in 
# SConscript file for any SIT project
Import('*')

from os.path import join as pjoin
from SConsTools.standardExternalPackage import standardExternalPackage

#
# For the standard external packages which contain includes, libraries, 
# and applications it is usually sufficient to call standardExternalPackage()
# giving some or all parameters.
#

pkg = "numpy"
pkg_ver = "1.9.2"

PREFIX = pjoin('$SIT_EXTERNAL_SW', pkg, pkg_ver)

libdir = 'lib'
PYDIR = pjoin(libdir, '$PYTHON', "site-packages", pkg)
PYDIRSEP = True
INCDIR = pjoin(PYDIR, "core", "include", pkg)
PKGINFO = (pkg, pkg_ver, '$PYTHON', '$SIT_ARCH.found')

standardExternalPackage(pkg, **locals())
