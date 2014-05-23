#!/usr/bin/env  python 


import InterfaceCtlr.FileMgrRegister


class IrodsRegisterFile:
    """ Register a file in irods """

    
    def __init__(self):
        self.no_register = False

    def do_not_register(self):
        self.no_register = True

    def register(self, path, instrument, experiment, ftype):
        
        print "IN REGISTER"
        if self.no_register:
            print "no i-register", path, instrument, experiment, ftype
            return
        
        print "i-register", path, instrument, experiment, ftype
        try:
            reg = InterfaceCtlr.FileMgrRegister.FileMgrRegister()
            reg.register(path, instrument, experiment, ftype)
        except:
            print "i-register failed", path

        

