#!/bin/sh
#+###########################################################
#
# sit_env.sh
#
# Description: Initial setup of SIT environment
#
# $Id$
#
#############################################################

SIT_ENV="default"; export SIT_ENV

tmp_sh=/tmp/uss-$$.sh
tmp_sit_root=${SIT_ROOT:-/reg/g/psdm}

$tmp_sit_root/bin/uss.sh -s $tmp_sit_root/etc/sit_env.uss > $tmp_sh && . $tmp_sh
/bin/rm $tmp_sh

unset tmp_sh
unset tmp_sit_root
