#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: template!SConscript! 8 2008-10-08 22:35:33Z salnikov $
#
# Description:
#  SConscript file for package nexus
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

boost_ver = "1.49.0"

PREFIX = pjoin(env['SIT_EXTERNAL_SW'], "boost", boost_ver + '-' + env['PYTHON'], env['SIT_ARCH'])
INCDIR = "include/boost"

# Mother of all other boost packages, this will only link 
# include directory into release
standardExternalPackage('boost', **locals())

# INCDIR needed any more
del INCDIR
LIBDIR = "lib"

# boost packages and their dependencies
pkgs = {'boost_chrono' : 'boost',
        'boost_date_time' : 'boost',
        'boost_filesystem' : 'boost_system boost',
        'boost_graph' : 'boost',
        'boost_iostreams' : 'boost',
        'boost_program_options' : 'boost',
        'boost_python' : 'boost python',
        'boost_random' : 'boost',
        'boost_regex' : 'boost',
        'boost_serialization' : 'boost',
        'boost_signals' : 'boost',
        'boost_system' : 'boost',
        'boost_thread' : 'boost',
        'boost_unit_test_framework' : 'boost',
        'boost_wave' : 'boost',
        'boost_wserialization' : 'boost',
        }
for pkg, dep in pkgs.iteritems():
    DEPS = dep
    PKGLIBS = pkg
    LINKLIBS = 'lib' + PKGLIBS + '*.so*'
    PKGINFO = None
    standardExternalPackage(pkg, **locals())
