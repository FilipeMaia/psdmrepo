#!/bin/bash 
#
# Display the disk usage of each ffb brick and send email 
#
# *mvr_showspaceffb* is used to get the df values. It uses 
# ssh (pdsh) to login the ffb host
#

. /reg/g/psdm/bin/sit_setup.sh dm-current


mailcmd=~wilko/repo/Utils/SendMail

tmpfile=/tmp/ffb_check_$(uuidgen)

mvr_showspaceffb all df 2>&1 | sort  &> ${tmpfile}
echo "Ran on $(hostname) $(date +'%Y%m%d %H:%M:%S')" >> ${tmpfile}

${mailcmd} -s "ffb disk space" -a wilko -f ${tmpfile}

[ -e ${tmpfile} ] && rm ${tmpfile}


