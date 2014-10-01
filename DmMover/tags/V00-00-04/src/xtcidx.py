
import os
import time
import errno
import shutil
import logging
import threading
import Queue
from datetime import datetime
from uuid import uuid4


# ========= helper functions =======================

def rename_tmp(tmpfn, fn):
    """ rename a tmpfile. If the target exists rename it first by adding a date and
    backup postfix.
    """

    if os.path.exists(fn):
        fn_bck = "%s.%s.%s" % (fn, datetime.now().strftime("%Y%m%dT%H%M%S"), "bck")
        logging.warning("mv %s %s", fn, fn_bck)
        shutil.move(fn, fn_bck)
        
    if not os.path.exists(fn):
        logging.info("mv %s %s", tmpfn, fn)
        shutil.move(tmpfn, fn)
    else:
        raise IOError("target files exists")


def create_idx(exp_file, config, logdir="/tmp"):
    """ Create a index file 
    exp_file is a object with attributes: xtc_dir, idx_file, instr, name
    config is a object with attribute: list_only 

    """
    
    xtc_dir, idx_name = exp_file.xtc_dir, exp_file.idx_file
    idx_path = os.path.join(xtc_dir, 'index', idx_name)

    uid = uuid4()
    idx_path_tmp = "%s.%s" % (idx_path, uid)

    logging.debug("idx files %s", idx_path)
        
    if os.path.exists(idx_path):
        logging.debug("index file exists %s", idx_path)
        return -1 

    xtc_path = os.path.join(xtc_dir, idx_name.replace('.idx',''))
    logging.debug("xtc file %s", xtc_path)
        
    if not os.path.exists(xtc_path):
        return -2

    outlog = os.path.join(logdir, "create_idx_%s_%s" % (uid, idx_name))
    cmd = "xtcindex -f %s -o %s > %s" % (xtc_path, idx_path_tmp, outlog) 
    logging.debug("listonly: %s cmd: %s", config.listonly, cmd)

    xtc_size = os.path.getsize(xtc_path)
    starttime = time.time()
    if config.listonly:
        return 0

    rc = os.system(cmd)
    if rc == 0:
        try:
            rename_tmp(idx_path_tmp, idx_path)
        except Exception:
            rc = errno.EIO

    elapstime = time.time() - starttime
    rate = float(xtc_size) / pow(2,20) / elapstime if elapstime > 0.01 else -1
 
    
    logging.info("Result %s %s %s elap %0.1f start %d bytes %d rate %0.1fMB/s rc %d", 
                 exp_file.instr, exp_file.name, idx_name, elapstime, int(starttime), 
                 xtc_size, rate, rc)
    return rc


# ================================================================================

class CreateIdx:

    def __init__(self, listonly=False):

        class Config:
            pass        
        self.cfg = Config()
        self.cfg.listonly = listonly
        self.logdir = "/tmp"

    def add_request(self, exp_file):
        return create_idx(exp_file, self.cfg, logdir=self.logdir)

    
