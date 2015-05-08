#!/usr/bin/env python 

import os
import logging
from socket import gethostname

class BbcpCmd(object):
    """ Transfer a file using bbcp
    
    Select bbcp options for specific transfer types.

    >>> mover = BbcpCmd()
    >>> mover.config_ffb_offline()
    >>> mover.transfer(src, src_path, trg, trg_path, options="extra bbcp options")
    """
    def __init__(self):     
        self.bbcp_remote = "ssh -x -a -oFallBackToRsh=no %I -l %U %H ionice -c2 -n7 /reg/g/pcds/pds/datamvr/bbcp-ssl" 
        self.bbcp = "/reg/g/pcds/pds/datamvr/bbcp-ssl"
        self.user, self.ssh_key = None, None
        self.local_transfer = False
        self.trg = gethostname()


    def config_local_offline(self):
        # src data read from a local file system (e.g.: nfs). Use realtime transfer.
        self.local_transfer = True
        self.bbcpcmd = "%s -v -P 15 -R c=2 -s 1" % self.bbcp

    def config_ffb_offline(self):
        # transfer from ffb to offline. No realtime as the file is complete 
        self.local_transfer = False
        self.bbcpcmd = "%s -S \"%s\"  -z -v -s 1 -p -P 15" % (self.bbcp, self.bbcp_remote)

    def config_dss_ffb(self):
        # transfer dss to ffb
        self.local_transfer = False
        self.bbcpcmd = "%s -S \"%s\"  -z -v -s 1 -R c=2 -P 15" % (self.bbcp, self.bbcp_remote)
        
    def print_config(self):
        print "Local-trans:", self.local_transfer, "bbcp-user:", self.user, " key:", self.ssh_key
        print "bbcp cmd:", self.bbcpcmd


    def to_local(self, src, src_path, trg_path, options=""):
        """ transfer a file to a local file system (no trg host)""" 
        cmd = self._cmd(src, src_path, None,  trg_path, options)
        logging.debug("bbcp cmd %s", cmd)
        return os.system(cmd)

    
    def _cmd(self, src, src_path, trg, trg_path, options=""):
        """ Create the bbcp command line """
        
        self.src, self.src_path = src, src_path
        self.trg_path = trg_path

        # format src url
        if self.local_transfer:
            srcurl = src_path
        else:
            if self.user:
                srcurl = "%s@%s:%s" % (self.user, src, src_path)
            elif src:
                srcurl = "%s:%s" % (src, src_path)
            else:
                srcurl = src_path

        # format target specification
        if trg:
            trgurl = "%s:%s" % (trg, trg_path)
        else:
            trgurl = trg_path


        if self.ssh_key:
            extra_options = "%s -i %s" % (options, self.ssh_key)
        else:
            extra_options = options

        return "%s %s %s %s" % (self.bbcpcmd, extra_options, srcurl, trgurl)

