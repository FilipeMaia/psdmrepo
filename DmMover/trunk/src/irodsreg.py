#!/usr/bin/env  python 


import InterfaceCtlr.FileMgrRegister


class IrodsRegisterFile:
    """ Register a file in irods """

    
    def __init__(self):
        self.no_register = set()

    def do_not_register(self, ftype):
        """ Set file type that should no be registered """
        self.no_register.add(ftype)

    def not_to_register(self):
        return " ".join(self.no_register)

    def register(self, path, instrument, experiment, ftype):
        
        print "IN REGISTER"
        if ftype in self.no_register:
            print "no i-register", path, instrument, experiment, ftype
            return
        
        print "i-register", path, instrument, experiment, ftype

        try:
            reg = InterfaceCtlr.FileMgrRegister.FileMgrRegister()
            reg.register(path, instrument, experiment, ftype)
        except:
            print "i-register failed", path

        

