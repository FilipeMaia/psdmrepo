

import os

cprefix = "/reg/d/psdm"

def iname2cdir(iname):
    """ return the canonical folder of a file extracted from 
    the irods data name.

    Example:
    /psdm-zone/psdm/XPP/xppi0813/xtc/e322-r0086-s04-c00.xtc ->
    /reg/d/psdm/xpp/xppi0813/xtc
    """

    path_token = iname.lstrip('/').split('/')
    return os.path.join(cprefix, path_token[2].lower(), *path_token[3:5])
    

class CheckTrgDir:
    """ Keep track if a directory exists """
    def __init__(self):
        self.dir_status = {}

    def status(self):
        for folder, stat in self.dir_status.iteritems():
            yield folder, stat

    def check_for_dir(self, dirpath):
        """ Check that path of the experiments sub folder exists and is a link.
        Keep track of already checked path. 
        """ 

        if dirpath in self.dir_status:
            return self.dir_status[dirpath]

        if os.path.exists(dirpath) and os.path.islink(dirpath):
            self.dir_status[dirpath] = True
        else:
            self.dir_status[dirpath] = False
            
        return self.dir_status[dirpath]
            
