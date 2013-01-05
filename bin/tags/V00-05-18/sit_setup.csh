#!/bin/csh
#+###########################################################
#
# sit_setup.csh
#
# description: Environment setup for SIT users
#
# $Id$
#
#######################################################################

if ( ${?SIT_SETUP_DIR} ) then
  set ussbindir="${SIT_SETUP_DIR}"
else if ( ${?SIT_ROOT} ) then
  set ussbindir="${SIT_ROOT}/bin"
else
  set ussbindir="/reg/g/psdm/bin"
endif
set tmp_sh=/tmp/uss-$$.csh

if ( -f $tmp_sh ) then
  /bin/rm $tmp_sh
endif
"$ussbindir/uss.sh" -c -- "$ussbindir/sit_setup.uss" ${argv:q} > $tmp_sh
source $tmp_sh

# cleanup
/bin/rm $tmp_sh
unset tmp_sh
unset ussbindir
