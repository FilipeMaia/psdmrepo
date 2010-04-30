#!/bin/sh
#+###########################################################
#
# lusi_setup.sh
#
# description: Environment setup for LUSI users
#
# $Id$
#
#######################################################################

scrdir="${LUSI_SETUP_DIR:-${LUSI_ROOT:-/reg/g/psdm}/bin}"
uss="$scrdir/lusi_setup.uss"
tmp_sh=/tmp/uss-$$.sh

test -f $tmp_sh && /bin/rm $tmp_sh
"$scrdir/uss.sh" -s -- "$uss" "$@" > $tmp_sh
. $tmp_sh
/bin/rm $tmp_sh
unset tmp_sh
unset scrdir