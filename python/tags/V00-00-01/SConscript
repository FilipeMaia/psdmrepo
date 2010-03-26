#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  SConscript file for package python
#------------------------------------------------------------------------

# Do not delete following line, it must be present in 
# SConscript file for any LUSI project
Import('*')

from SConsTools.standardExternalPackage import standardExternalPackage

#
# For the standard external packages which contain includes, libraries, 
# and applications it is usually sufficient to call standardExternalPackage()
# giving some or all parameters.
#

python_ver = env['PYTHON_VERSION']
PREFIX  = "/usr"
INCDIR = "include/python"+python_ver
LIBDIR  = "lib"
if env['LUSI_ARCH_PROC'] == 'x86_64' : LIBDIR  = "lib64"
PKGLIBS = "python"+python_ver
LINKLIBS = "libpython"+python_ver+".so*"

standardExternalPackage ( 'python', **locals() )
