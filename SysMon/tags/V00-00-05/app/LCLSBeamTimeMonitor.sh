#!/bin/sh

# Translate parameters of the sampler if provided. Otherwise assume
# their default values.
#
SAMPLE_INTERVAL_SEC=1.0
FORCE_INTERVAL_SEC=10.0

if ! [[ -z $1 ]] ; then
    SAMPLE_INTERVAL_SEC=$1
fi
if ! [[ -z $2 ]] ; then
    FORCE_INTERVAL_SEC=$2
fi

# Make sure the script is runing on the right host
#
HOST2RUN=psdev101
if [ `hostname` != "$HOST2RUN" ] ; then
  echo "error: the script should only be run on $HOST2RUN"
  exit 1
fi

# Verify and set up execution environment
#
if [ ! -r "/reg/g/pcds/setup/pyca.sh" ]; then
  echo "error: no PYCA setup found"
  exit 1
fi
source /reg/g/pcds/setup/pyca.sh

if [ ! "`which python`" ]; then
  echo "error: no Python interpreter available in $PATH"
  exit 1
fi
if ! python -V 2>&1 | /bin/egrep -q 'Python 2.5' ; then
  echo "error: the script requires Python 2.5.*"
  exit 1
fi

if [[ -z "$PYTHONPATH" ]] ; then
    export PYTHONPATH=/reg/g/pcds/pds/sysmon/lib/python2.5/site-packages/
else
    export PYTHONPATH=$PYTHONPATH:/reg/g/pcds/pds/sysmon/lib/python2.5/site-packages/
fi

# Check that all scripts needed by the Monitor are
# available.
#
BASEDIR=`dirname $0`/

WATCH=LCLSBeamTimeWatch.py
if [ ! -x $BASEDIR/$WATCH ] ; then
    echo "error: $WATCH is not found in $BASEDIR"
    exit 1
fi
STORE=LCLSBeamTimeStore.py
if [ ! -x $BASEDIR/$STORE ] ; then
    echo "error: $STORE is not found in $BASEDIR"
    exit 1
fi

# Finally, proceed to the actaul monitoring tool. When it finishes
# return its status up stream.
#
$BASEDIR/$WATCH $SAMPLE_INTERVAL_SEC $FORCE_INTERVAL_SEC

exit $?
