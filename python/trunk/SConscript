#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  SConscript file for package python
#------------------------------------------------------------------------

# Do not delete following line, it must be present in 
# SConscript file for any SIT project
Import('*')

from SConsTools.standardExternalPackage import standardExternalPackage

#
# For the standard external packages which contain includes, libraries, 
# and applications it is usually sufficient to call standardExternalPackage()
# giving some or all parameters.
#

python = env['PYTHON']   # python with version number such as python2.7
PREFIX = "/usr"
INCDIR = env['PYTHON_INCDIR']
LIBDIR  = env['PYTHON_LIBDIR']
PKGLIBS = python
LINKLIBS = "lib"+python+".so*"

standardExternalPackage ( 'python', **locals() )
