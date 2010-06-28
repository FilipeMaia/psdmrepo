#!/bin/csh
#+###########################################################
#
# sit_env.csh
#
# Description: Initial setup of SIT environment
#
# $Id$
#
#############################################################

setenv SIT_ENV "default"

set tmp_sh=/tmp/uss-$$.csh

/reg/g/psdm/bin/uss.sh -c /reg/g/psdm/etc/sit_env.uss > $tmp_sh
source $tmp_sh
rm $tmp_sh
unset tmp_sh
