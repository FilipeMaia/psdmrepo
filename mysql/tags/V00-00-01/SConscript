#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: SConscript 184 2009-01-31 08:22:52Z salnikov $
#
# Description:
#  SConscript file for package unixodbc
#------------------------------------------------------------------------

# Do not delete following line, it must be present in 
# SConscript file for any LUSI project
Import('*')

import os
from os.path import join as pjoin

from SConsTools.standardExternalPackage import standardExternalPackage

#
# For the standard external packages which contain includes, libraries, 
# and applications it is usually sufficient to call standardExternalPackage()
# giving some or all parameters.
#

#unixodbc_ver = "2.2.12"

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
LIBDIR  = "lib/mysql"
PKGLIBS = "mysqlclient"
LINKLIBS = "libmysqlclient.so*"
standardExternalPackage ( 'mysql', **locals() )
