#!/usr/bin/env  python 

import logging
import InterfaceCtlr.FileMgrRegister


class IrodsRegisterFile:
    """ Register a file in irods. 

    Needs to specify file types that should be registered
    """

    def __init__(self):
        self.to_register = set()

    def register_ftype(self, ftype):
        """ Set file type that should be registered """
        self.to_register.add(ftype)

    def list_to_register(self):
        return " ".join(self.to_register)

    def register(self, path, instrument, experiment, ftype):
        
        if ftype not in self.to_register:
            logging.warning("no i-register %s %s %s %s", path, instrument, experiment, ftype)
            return
        
        try:
            reg = InterfaceCtlr.FileMgrRegister.FileMgrRegister()
            reg.register(path, instrument, experiment, ftype)
        except:
            logging.error("i-register failed %s", path)
        else:
            logging.info("i-register %s %s %s %s", path, instrument, experiment, ftype)

        
            
        

