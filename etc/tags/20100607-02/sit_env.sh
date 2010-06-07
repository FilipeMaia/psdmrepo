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

/reg/g/psdm/bin/uss.sh -s /reg/g/psdm/etc/sit_env.uss > $tmp_sh
. $tmp_sh
rm $tmp_sh
unset tmp_sh
