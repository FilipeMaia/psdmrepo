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

# Python configuration is determined by SConsTools, check SCons tools psdm_python

python = env['PYTHON']   # python with version number such as python2.7
PREFIX = env['PYTHON_PREFIX']
INCDIR = env['PYTHON_INCDIR']
LIBDIR  = env['PYTHON_LIBDIR']
BINDIR  = env['PYTHON_BINDIR']
PKGLIBS = python
LINKLIBS = ["lib"+python+".so*", 'libtcl8.5.so*', 'libtk8.5.so*']
LINKBINS = ["python", python]

standardExternalPackage('python', **locals())
