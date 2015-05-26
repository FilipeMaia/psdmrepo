
import os
import subprocess

def files_in_queue():
    """ Return the irods data names that are queued """

    cmd = "/reg/neh/home1/psdatmgr/bin/iqstatw"

    res = subprocess.check_output(cmd)
    in_queue = set( tok.split()[-1] for tok in res.splitlines() if len(tok) > 1 )
    return in_queue


(ST_DISK, ST_RESTORE, ST_MISS) = (0,1,2)

def file_on_disk(iname):
    """ Check if a irods data object is accessible on disk. 
    check for the file name or the temp name that is used for 
    restoring 
    """
    
    regpath = os.path.join("/reg/d/psdm/", iname[psdm_prefix:])
    
    if os.path.exists(regpath):
        return ST_DISK
    elif os.path.exists("%s.fromtape" % regpath):
        return ST_RESTORE
    else:
        return ST_MISS

        
