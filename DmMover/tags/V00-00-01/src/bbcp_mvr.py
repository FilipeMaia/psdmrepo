#!/usr/bin/env python 


class BbcpCmd:

    def __init__(self):     
        self.bbcp_remote = "ssh -x -a -oFallBackToRsh=no %I -l %U %H ionice -c2 -n7 /reg/g/pcds/pds/datamvr/bbcp-ssl" 
        self.bbcp = "/reg/g/pcds/pds/datamvr/bbcp-ssl"
        self.user, self.ssh_key = None, None
        self.local_transfer = False
        
    def config_local_offline(self):
        # src data read from a local file system (e.g.: nfs), Use realtime transfer
        self.local_transfer = True
        self.bbcpcmd = "%s -v -P 15 -R c=2 -s 1" % self.bbcp

    def config_ffb_offline(self):
        # transfer from ffb to offline. No realtime as the file is complete 
        self.bbcpcmd = "%s -S \"%s\"  -z -v -s 1 -p -P 15" % (self.bbcp, self.bbcp_remote)
        
    def print_config(self):
        print "Local-trans:", self.local_transfer, "bbcp-user:", self.user, " key:", self.ssh_key
        print "bbcp cmd:", self.bbcpcmd
    
    def cmd(self, srchost, src, trg, options=""):
        """ Create the bbcp command line argument """

        # local file to file copy
        if self.local_transfer:
            return "%s %s %s %s" % (self.bbcpcmd, options, src, trg)
           
        if self.user:
            srcurl = "%s@%s:%s" % (self.user, srchost, src)
        else:
            srcurl = "%s:%s" % (srchost, src)

        if self.ssh_key:
            extra_options = "%s -i %s" % (options, self.ssh_key)
        else:
            extra_options = options

        return "%s %s %s %s" % (self.bbcpcmd, extra_options, srcurl, trg)
    


if __name__ == "__main__":

    import os
    b = BbcpCmd()
    b.config_ffb()
    b.print_config()
    cmd = b.cmd('fgt', '/reg/d/cameras/ioc-xrt-xcscam02/wilko/f6', '/reg/data/ana01/test/wilko/f6')
    print cmd
    print " "
    b.config_local()
    b.print_config()
    cmd = b.cmd('fgt', '/reg/d/cameras/ioc-xrt-xcscam02/wilko/f6', '/reg/data/ana01/test/wilko/f6')
    print cmd
    
    

    #rc = os.system(cmd)
    #print rc
