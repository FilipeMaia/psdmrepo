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

if ( ${?LUSI_SETUP_BIN} ) then
  set scrdir="${LUSI_SETUP_BIN}"
else if ( ${?LUSI_ROOT} ) then
  set scrdir="${LUSI_ROOT}/bin"
else
  set scrdir="/reg/g/psdm/bin"
endif
set uss="$scrdir/lusi_setup.uss"
set tmp_sh=/tmp/uss-$$.csh

if ( -f $tmp_sh ) then
  /bin/rm $tmp_sh
endif
"$scrdir/uss.sh" -c -- "$uss" ${argv:q} > $tmp_sh
source $tmp_sh
/bin/rm $tmp_sh
unset tmp_sh
unset scrdir
