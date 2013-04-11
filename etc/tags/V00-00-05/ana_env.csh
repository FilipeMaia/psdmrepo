#!/bin/csh
#+###########################################################
#
# ana_env.csh
#
# Description: Initial setup of SIT environment
#
# $Id$
#
#############################################################

setenv SIT_ENV "ana"

set tmp_sh=/tmp/uss-$$.csh
set tmp_sit_root='/reg/g/psdm'
if ( ${?SIT_ROOT} ) then
  set tmp_sit_root=${SIT_ROOT}
endif

$tmp_sit_root/bin/uss.sh -c $tmp_sit_root/etc/sit_env.uss > $tmp_sh
source $tmp_sh
rm $tmp_sh

unset tmp_sh
unset tmp_sit_root
