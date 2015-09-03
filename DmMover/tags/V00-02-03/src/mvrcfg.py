#!/usr/bin/env python 

import os
import time
import ConfigParser
import argparse
import psutil
import socket

from datetime import datetime


# ====================================================================================


def which(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

# ====================================================================================

class MoverCfg(object):
    """ Configuration for a mover """

    def __init__(self, name):
        self.name = name
        self.opts = ""
        self.sel = None
        self.smd = False
        self.release = None 

    def set_config(self, cfg):

        for opt,val in cfg:
            if opt == "mode":
                self.mode = "--mode {}".format(val)
            else:
                self.__setattr__(opt, val)

            if not self.sel:
                self.sel = "--host {}".format(self.name) 
            
    def procserv_logfile(self):
        return "{0.logdir}/mv2offline.{0.name}".format(self)

    def mvr_cmd(self):

        cmd_abspath = which(self.cmd)
        cmd = (
            "{0.procserv_cmd} {0.procserv_args} "
            "-L {0.logdir}/mv2offline.{0.name} -n {0.name} "
            "{0.port} "
            "{1} {0.sel} {0.mode} {0.options}".format(self, cmd_abspath) 
            )
        return cmd 


# ====================================================================================


class Config(object):
    """ Read mover config from a ini style file """

    def __init__(self, cfgfile, only_localhost=True):

        self._movers = []

        config = ConfigParser.SafeConfigParser()
        config.read(cfgfile) 
        
        ports = [ config.get(section,'port') for section in config.sections()]
        if len(ports) != len(set(ports)):
            print "Found duplicate ports", ports
            return

        localhost = socket.gethostname()    
        for section in config.sections():
            if not config.has_option(section, "host"):
                print "Missing host in section", section
                continue
                
            if only_localhost and localhost != config.get(section, "host"):
                continue

            mvr = MoverCfg(section)
            mvr.set_config(config.items(section))
            self._movers.append(mvr)


    def movers(self):
        for mvr in self._movers:
            yield mvr

