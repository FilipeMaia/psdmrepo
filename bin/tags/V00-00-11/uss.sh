#!/bin/sh
#
# $Id$
#
#   uss.sh - universal scripting solution
#
#   from bourne-based shells:
#       f=/tmp/uss-$$.sh
#       uss.sh -s uss_script > $f
#       . $f
#       rm $f
#
#   from csh-based shells:
#       set f=/tmp/uss-$$.sh
#       uss.sh -c uss_script > $f
#       . $f
#       rm $f
#

usage() {
  cat << EOD
Usage: $0 [options] uss_script [scrip_options]
  Available options:
    -h         this message
    -c         generate csh code
    -s         generate sh code (default)
EOD
}

uss_mode="sh"

# get the options
while getopts hcs c ; do
  case $c in
    h) usage ; exit 0 ;;
    c) uss_mode="csh" ;;
    s) uss_mode="sh" ;;
    \?) usage ; exit 2 ;;
  esac
done
shift `expr $OPTIND - 1`

# check arguments
if [ $# -eq 0 ] ; then 
  usage
  exit 2
fi

# =============================
#   Few common shell functions
# =============================
_remove_from_path ()
{ 
  # 1. split original path at : into separate lines
  # 2. filter out matching lines and empty lines
  # 3. replace newlines with :
  # 4. strip trailing :
  #
  echo "$1" | tr ':' '\n'| sed -e "\\:^$2\$:d" -e '/^ *$/d' | tr '\n' ':' | sed -e 's/\(.*\):/\1/'
}


# =============================
#   Bourne shell functions
# =============================
if [ "$uss_mode" = "sh" ] ; then 

set_env ()
{ 
  eval $1=\"$2\"
  export $1
  echo $1=\"$2\"
  echo export $1
}

unset_env ()
{ 
  unset $1
  echo unset $1
}

prepend_path ()
{
  eval uss_path=\$$1
  if [ -z "$uss_path" ] ; then
    set_env $1 "$2"
  else
    eval $1=\"$2:$uss_path\"
    export $1
    echo $1=\"$2:$uss_path\"
    echo export $1
  fi
  unset uss_path
}

append_path  ()
{
  eval uss_path=\$$1
  if [ -z "$uss_path" ] ; then
    set_env $1 "$2"
  else
    eval $1=\"$uss_path:$2\"
    export $1
    echo $1=\"$uss_path:$2\"
    echo export $1
  fi
  unset uss_path
}

remove_from_path ()
{ 
  eval uss_path=\$$1
  uss_path=`_remove_from_path "$uss_path" "$2"`
  eval $1=\"$uss_path\"
  export $1
  echo $1=\"$uss_path\"
  echo export $1
  unset uss_path
}

set_alias ()
{ 
  alias $1="$2"
  echo alias $1=\"$2\"
}

unset_alias ()
{ 
  unalias $1
  echo unalias $1
}

fi

# =============================
#   C-shell functions
# =============================
if [ "$uss_mode" = "csh" ] ; then 

set_env ()
{ 
  eval $1=\"$2\"
  export $1
  echo setenv $1 \"$2\"
}

unset_env ()
{ 
  unset $1
  echo unsetenv $1
}

prepend_path ()
{
  eval uss_path=\$$1
  if [ -z "$uss_path" ] ; then
    set_env $1 "$2"
  else
    eval $1=\"$2:$uss_path\"
    export $1
    echo setenv $1 \"$2:$uss_path\"
  fi
  unset uss_path
}

append_path  ()
{
  eval uss_path=\$$1
  if [ -z "$uss_path" ] ; then
    set_env $1 "$2"
  else
    eval $1=\"$uss_path:$2\"
    export $1
    echo setenv $1 \"$uss_path:$2\"
  fi
  unset uss_path
}

remove_from_path ()
{ 
  eval uss_path=\$$1
  uss_path=`_remove_from_path "$uss_path" "$2"`
  eval $1=\"$uss_path\"
  export $1
  echo setenv $1 \"$uss_path\"
  unset uss_path
}

set_alias ()
{ 
  alias $1="$2"
  echo alias $1 \"$2\"
}

unset_alias ()
{ 
  unalias $1
  echo unalias $1
}

fi


# ======================
#  Now source the beast
# ======================
uss_script="$1"
shift
. "$uss_script"

