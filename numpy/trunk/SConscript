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
pkg_ver = "1.3.0"

PREFIX = pjoin(env['SIT_EXTERNAL_SW'], pkg, pkg_ver)

PYDIR = pjoin(env['LIB_ABI'], env['PYTHON'], "site-packages", pkg)
PYDIRSEP = True
INCDIR = PYDIR + "/core/include/" + pkg
PKGINFO = (pkg, pkg_ver, env['PYTHON'], env['SIT_ARCH'])

standardExternalPackage(pkg, **locals())
