#! /bin/sh
#+###########################################################
#
# group_sys.conf.sh
#
# description: Environment for LUSI users
#
# $Id$
#
#######################################################################
if [ -z "$GROUP_ENV" ]; then

  uss=/reg/g/psdm/etc/hepix/group_sys.conf.uss
  tmp_sh=/tmp/uss-$$.sh

  /reg/g/psdm/bin/uss.sh -s $uss > $tmp_sh
  . $tmp_sh
  rm $tmp_sh

  GROUP_ENV="LUSI"; export GROUP_ENV

fi
