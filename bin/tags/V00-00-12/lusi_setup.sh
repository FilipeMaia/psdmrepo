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

uss=/afs/slac.stanford.edu/g/lusi/bin/lusi_setup.uss
tmp_sh=/tmp/uss-$$.sh

test -f $tmp_sh && /bin/rm $tmp_sh
/afs/slac.stanford.edu/g/lusi/bin/uss.sh -s $uss "$@" > $tmp_sh
. $tmp_sh
/bin/rm $tmp_sh
unset tmp_sh
