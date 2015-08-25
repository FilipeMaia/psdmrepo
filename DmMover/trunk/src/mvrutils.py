

import os
import logging

# logger for this module 
logger = logging




def network_name(hostname):
    """ Translate the src hostname into a name that is used by the transfer
    
    Only for dss -> ffb transferes the name has to be translated.
    """

    if hostname.startswith('daq-'):
        return "10.1.1.1"

    if hostname.startswith('ioc-'):
        return "psana-" + hostname[4:]
        
    #if hostname == "ioc-fee-rec02":
    #    return "psana-fee-rec02"
    
    return hostname


# ============================================================================================


def trgpath_param(mode, path=None):
    """ Return a function that returns a tuple with status
             (register_to_ana_tb, create_xtc_only, instr_in_trg_path, rootpath)
    for an instrument. The rootpath is used to write files e.g.: <rootpath>/<instr>/....
    The rootpath is None if the path (datapath) from the DB should be used.
    If register_to_ana is True the transfer will be registered to the data_migration
    table. If instr_in_trg_path=True the target path will include the instrument name (default)
    as shown above. For dss->ffb transfers the instr must not be included.

    >>> fn = trgpath_param("ioc-ffb", "/reg/data/ana12")
    >>> reg_ana, incl_instr, rootpath = fn("cxi") 
    """
    
    def rootpath(instr):
        """ Return  (reg_ana, cr_instr, rpath)  """
        
        #  (register_to_ana_tb, create_xtc_only, instr_in_trg_path, rootpath)
        if mode == "dss-ffb" or mode == "dsslocal-ffb":
            return (True, True, False, path)
        elif mode in ("ioc-ffb", "ioclocal-ffb"):
            if instr.upper() in ('AMO', 'CXI', 'MEC', 'SXR', 'XCS', 'XPP'):
                return (True, True, True, path)
            else:
                # dia experiments
                return (False, False, True, None)
        
        return (False, False, True, path)
    
    return rootpath


# ========================================================================================


def rm_remote_file(rhost, rfile, lfile, remote_cmd):
    """ remove rfile on rhost checking that the file sizes
    of rfile and lfile match. Execute shell command on rhost
    with ssh.
    """
    
    try:
        size = os.path.getsize(lfile)
    except OSError:
        logging.error("rm-file no local file %s", lfile)
        return False
    
    cmd = "ssh -x %s %s %s %d %s" % (rhost, remote_cmd, rfile, size, gethostname())
    print cmd
    rc = os.system(cmd)
    if rc != 0:
        logging.error("remove failed %s", lfile)
        return False
    return True



