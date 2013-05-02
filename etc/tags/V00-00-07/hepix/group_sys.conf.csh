#! /bin/csh
#+###########################################################
#
# group_sys.conf.csh
#
# description: Environment for LUSI users
#
# $Id$
#
#######################################################################
if ( ! $?GROUP_ENV ) then

  set uss=/reg/g/psdm/etc/hepix/group_sys.conf.uss
  set tmp_sh=/tmp/uss-$$.csh

  /reg/g/psdm/bin/uss.sh -c $uss > $tmp_sh
  source $tmp_sh
  rm $tmp_sh

  setenv GROUP_ENV "LUSI"

endif
