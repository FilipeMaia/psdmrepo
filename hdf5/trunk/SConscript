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

from SConsTools.standardExternalPackage import standardExternalPackage

#
# For the standard external packages which contain includes, libraries, 
# and applications it is usually sufficient to call standardExternalPackage()
# giving some or all parameters.
#

hdf5_ver = "1.8.1"
hdf5_dir = os.path.join(env['LUSI_ROOT'],"sw/external/hdf5",hdf5_ver,env['LUSI_ARCH'])


standardExternalPackage ( 'hdf5', prefix=hdf5_dir, 
                          inc_dir="include", lib_dir="lib", bin_dir="bin" )


