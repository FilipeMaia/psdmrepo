#! /bin/csh
#+###########################################################
#
# lusi_setup.csh
#
# description: Environment setup for LUSI users
#
# $Id$
#
#######################################################################

set uss=/afs/slac.stanford.edu/g/lusi/bin/lusi_setup.uss
set tmp_sh=/tmp/uss-$$.csh

if ( -f $tmp_sh ) then
  /bin/rm $tmp_sh
endif
/afs/slac.stanford.edu/g/lusi/bin/uss.sh -c "$uss" ${argv:q} > $tmp_sh
source $tmp_sh
/bin/rm $tmp_sh
unset tmp_sh

