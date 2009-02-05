#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: template!SConscript! 8 2008-10-08 22:35:33Z salnikov $
#
# Description:
#  SConscript file for package nexus
#------------------------------------------------------------------------

# Do not delete following line, it must be present in 
# SConscript file for any LUSI project
Import('*')

import os
from os.path import join as pjoin

from SConsTools.standardExternalPackage import standardExternalPackage

#
# Find out suffix for boost libs
#
libsfxs = { 
    'g++34' : '-gcc34-mt',
    'g++-3.4' : '-gcc34-mt',
    'g++41' : '-gcc41-mt',
    'g++-4.1' : '-gcc41-mt',
    'g++' : '-gcc41-mt',
}
libsfx = libsfxs.get ( env['CXX'], '' )

#
# For the standard external packages which contain includes, libraries, 
# and applications it is usually sufficient to call standardExternalPackage()
# giving some or all parameters.
#

boost_ver = "1.36.0"

PREFIX  = pjoin(env['LUSI_ROOT'],"sw/external/boost",boost_ver,env['LUSI_ARCH'])
INCDIR  = "include/boost-1_36/boost"

# Mother of all other boost packages, this will only link 
# include directory into release
standardExternalPackage ( 'boost', **locals() )

# INCDIR needed any more
del INCDIR
LIBDIR = "lib"

# boost packages and their dependencies
pkgs = {'boost_date_time' : 'boost', 
        'boost_filesystem' : 'boost', 
        'boost_iostreams' : 'boost',
        'boost_regex' : 'boost',
        'boost_thread' : 'boost',
        'boost_unit_test_framework' : 'boost', 
        }
for pkg, dep in pkgs.iteritems() :
    DEPS = dep
    PKGLIBS = pkg+libsfx
    LINKLIBS = 'lib'+PKGLIBS+'*.so*' 
    standardExternalPackage ( pkg, **locals() )
