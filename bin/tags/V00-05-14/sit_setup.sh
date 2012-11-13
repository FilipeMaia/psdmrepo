#!/bin/sh
#+###########################################################
#
# sit_setup.sh
#
# description: Environment setup for SIT users
#
# $Id$
#
#######################################################################

ussbindir="${SIT_SETUP_DIR:-${SIT_ROOT:-/reg/g/psdm}/bin}"
tmp_sh=/tmp/uss-$$.sh

test -f $tmp_sh && /bin/rm $tmp_sh
"$ussbindir/uss.sh" -s -- "$ussbindir/sit_setup.uss" "$@" > $tmp_sh
. $tmp_sh

# cleanup
/bin/rm $tmp_sh
unset tmp_sh
unset ussbindir
