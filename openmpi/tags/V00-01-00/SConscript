#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: SConscript 3589 2012-05-22 14:54:40Z salnikov@SLAC.STANFORD.EDU $
#
# Description:
#  SConscript file for package openmpi
#------------------------------------------------------------------------

# Do not delete following line, it must be present in 
# SConscript file for any SIT project
Import('*')

from os.path import join as pjoin
from SConsTools.standardExternalPackage import standardExternalPackage

import os

#
# For the standard external packages which contain includes, libraries, 
# and applications it is usually sufficient to call standardExternalPackage()
# giving some or all parameters.
#

pkg = 'openmpi'
pkg_ver = '1.8.6'

PREFIX = '/'.join(["$SIT_ROOT", "sw", "external", pkg, pkg_ver, "$SIT_ARCH"])

INCDIR = "include"

LIBDIR = "lib"
LINKLIBS = "lib*.so*"

PKGLIBS = "mca_common_sm mpi mpi_cxx mpi_mpifh mpi_usempi ompitrace open-pal open-rte open-trace-format oshmem otfaux"

BINDIR = "bin"
LINKBINS = "*"

standardExternalPackage(pkg, **locals())

# Post fix up is needed to set up proper include directories
# and headers

env = DefaultEnvironment()

geninc = str(Dir(env.subst("$ARCHINCDIR")))

openmpiinc = pjoin(geninc, pkg)

if os.path.islink(openmpiinc):    os.remove(openmpiinc)
if not os.path.isdir(openmpiinc): os.makedirs(openmpiinc)

prefix = pjoin(os.path.dirname(str(Dir(env.subst(PREFIX)))), env['SIT_ARCH_BASE_OPT'])

s_mpi_h         = pjoin(prefix, "include", "mpi.h")
geninc_t_mpi_h  = pjoin(geninc, "mpi.h")
openmpi_t_mpi_h = pjoin(openmpiinc, "mpi.h")

if not os.path.islink(geninc_t_mpi_h):  os.symlink(s_mpi_h, geninc_t_mpi_h)
if not os.path.islink(openmpi_t_mpi_h): os.symlink(s_mpi_h, openmpi_t_mpi_h)

s_mpi_portable_platform_h        = pjoin(prefix, "include", "mpi_portable_platform.h")
geninc_t_mpi_portable_platform_h = pjoin(geninc, "mpi_portable_platform.h")
if not os.path.islink(geninc_t_mpi_portable_platform_h): os.symlink(s_mpi_portable_platform_h, geninc_t_mpi_portable_platform_h)

s_ompi         = pjoin(prefix, "include", "openmpi", "ompi")
openmpi_t_ompi = pjoin(openmpiinc, "ompi")

if not os.path.islink(openmpi_t_ompi): os.symlink(s_ompi, openmpi_t_ompi)

