#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: SConscript 184 2009-01-31 08:22:52Z salnikov $
#
# Description:
#  SConscript file for package unixodbc
#------------------------------------------------------------------------

# Do not delete following line, it must be present in 
# SConscript file for any SIT project
Import('*')

import os
from os.path import join as pjoin
import subprocess

from SConsTools.standardExternalPackage import standardExternalPackage

#
# For the standard external packages which contain includes, libraries, 
# and applications it is usually sufficient to call standardExternalPackage()
# giving some or all parameters.
#

PREFIX  = "/usr"
INCDIR  = "include/mysql"
INCLUDES = " ".join((
  "chardefs.h",
  "decimal.h",
  "errmsg.h",
  "history.h",
  "keycache.h",
  "keymaps.h",
  "m_ctype.h",
  "m_string.h",
  "my_*.h",
  "mysql*.h",
  "raid.h",
  "readline.h",
  "rlmbutil.h",
  "rlprivate.h",
  "rlshell.h",
  "rltypedefs.h",
  "sql_common.h",
  "sql_state.h",
  "sslopt-case.h",
  "sslopt-longopts.h",
  "sslopt-vars.h",
  "tilde.h",
  "typelib.h",
  "xmalloc.h" ))
LIBDIR  = "$LIB_ABI/mysql"
PKGLIBS = "mysqlclient"
LINKLIBS = "libmysqlclient.so*"

standardExternalPackage ( 'mysql', **locals() )

#
# very special target which makes libmysql_soname.h header 
#
def make_libmysql_soname_header(env, target, source):
    
    subp = subprocess.Popen(["objdump", "-p", str(source[0])], stdout=subprocess.PIPE)
    soname = "libmysqlclients.so"
    for line in subp.stdout:
        words = line.split()
        if len(words) == 2 and words[0] == "SONAME":
            soname = words[1]
    trgt = open(str(target[0]), "w")
    print >> trgt, 'const char* const libmysql_soname = "%s";' % soname
    
header = "#arch/$SIT_ARCH/geninc/mysql/libmysql_soname.h"
lib = pjoin(PREFIX, LIBDIR, "libmysqlclient.so")
target = env.Command(header, lib, make_libmysql_soname_header)
env['ALL_TARGETS']['INCLUDES'].append(target)
