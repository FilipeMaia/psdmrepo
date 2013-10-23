#
# $Id$
#
# Copyright (c) 2010 SLAC National Accelerator Laboratory
# 

import logging
import types

from pypdsdata import xtc
import pyana

_log = logging.getLogger("pyana.userana")

#
# Bunch of special methods to augment use module with
#

_default = [[[((()))]]]  # unlikely value for a default

_trues = {'yes': True, 'true': True, 'on': True, '1': True,
          'no': False, 'false': False, 'off': False, '0':False }
def _val2bool(val):
    val = _trues.get(val.lower())
    if val is None: raise ValueError("string cannot be converted to boolean: "+val)
    return val

def _configStr(self, parm, default=_default):
    if default is _default:
        return self.__pyana_config__[parm]
    else:
        return self.__pyana_config__.get(parm, default)

def _configBool(self, parm, default=_default):
    if parm in self.__pyana_config__: return _val2bool(self.__pyana_config__[parm])
    if default is _default: raise KeyError(parm)
    return default

def _configInt(self, parm, default=_default):
    if parm in self.__pyana_config__: return int(self.__pyana_config__[parm])
    if default is _default: raise KeyError(parm)
    return default

def _configFloat(self, parm, default=_default):
    if parm in self.__pyana_config__: return float(self.__pyana_config__[parm])
    if default is _default: raise KeyError(parm)
    return default

def _configSrc(self, parm, default=_default):
    # returns the string
    return _configStr(self, parm, default)


def _configListStr(self, parm):
    str = self.__pyana_config__.get(parm, "")
    return str.split()

def _configListBool(self, parm):
    return [_val2bool(x) for x in _configListStr(self, parm)]

def _configListInt(self, parm):
    return [int(x) for x in _configListStr(self, parm)]

def _configListFloat(self, parm):
    return [float(x) for x in _configListStr(self, parm)]

def _configListSrc(self, parm):
    return _configListStr(self, parm)

def _modName(self):
    return self.__modname__

def _className(self):
    return self.__modname__.split(':')[0]



def mod_import(name, config):
    """ Helper function to import and instantiate user analysis class """

    modname = name.split(':')[0]
    
    # import module
    try :
        mod = __import__(modname)
    except ImportError, e :
        _log.error("Cannot import module %s: %s", modname, str(e) )
        return None

    # locate sub-module
    components = modname.split('.')
    for comp in components[1:]:
        mod = getattr(mod, comp)
    
    # must define class with the same name
    classname = components[-1]
    userClass = getattr(mod, classname, None)
    if not userClass :
        _log.error("User module %s does not define class %s", modname, classname )
        return None
    
    # add bunch of methods
    setattr(userClass, 'configStr', _configStr)
    setattr(userClass, 'configBool', _configBool)
    setattr(userClass, 'configInt', _configInt)
    setattr(userClass, 'configFloat', _configFloat)
    setattr(userClass, 'configSrc', _configSrc)
    setattr(userClass, 'configListStr', _configListStr)
    setattr(userClass, 'configListBool', _configListBool)
    setattr(userClass, 'configListInt', _configListInt)
    setattr(userClass, 'configListFloat', _configListFloat)
    setattr(userClass, 'configListSrc', _configListSrc)
    setattr(userClass, 'name', _modName)
    setattr(userClass, 'className', _className)
    
    # temporary hack to access config dict in constructor
    configDict = config.copy()
    setattr(userClass, '__pyana_config__', configDict)
    setattr(userClass, "__modname__", name)
    
    # instantiate it
    userana = None
    err = None
    try:
        userana = userClass(**config)
    except TypeError, e :
        err = str(e)
        # possibly due to constructor not taking any arguments, try again without it
        try:
            userana = userClass()
        except Exception, e :
            err = str(e)
            pass
    except Exception, e :
        err = str(e)
        pass
    if userana is None:
        _log.exception("Failure while instantiating class %s: %s", classname, err )
        return None

    delattr(userClass, '__pyana_config__')
    delattr(userClass, '__modname__')

    # some weird cases are possible, just make sure that callable
    # returns an object which has a list of attributes that we need
    if not userana :
        _log.error("Failure while calling %s(): None returned", classname, str(e) )
        return None

    for method in ['beginjob', 'event', 'endjob' ] :
        if not hasattr(userana, method) :
            _log.error("User analysis class %s does not define method %s", classname, method )
            return None

    # make an instance attribute of the config dict
    setattr(userana, '__pyana_config__', configDict)
    setattr(userana, "__modname__", name)

    # looks OK so far
    return userana



class evt_dispatch(object) :
    
    def __init__ (self, userObjects) :
        self.userObjects = userObjects
        self.lastConfigTime = None
        self.runbegun = False
        self.calibbegun = False
        
    def dispatch (self, evt, env):
    
        svc = evt.seq().service()
        
        # process all data
        if svc == xtc.TransitionId.Configure :
            
            # run configure if it was not run or we have a different 
            # configure transition 
            cfgTime = evt.getTime()
            if cfgTime != self.lastConfigTime :
                if not self.lastConfigTime :
                    for userana in self.userObjects : userana.beginjob( evt, env )
                else:
                    _log.warning("Multiple Configure transitions encountered.")
                self.lastConfigTime = cfgTime
            
        elif svc == xtc.TransitionId.BeginRun :
            
            for userana in self.userObjects :
                if hasattr(userana, 'beginrun'):
                    userana.beginrun( evt, env )
            self.runbegun = True
            
        elif svc == xtc.TransitionId.EndRun :
            
            for userana in self.userObjects : 
                if hasattr(userana, 'endrun'):
                    meth = userana.endrun
                    if meth.func_code.co_argcount == 3:
                        meth( evt, env )
                    else:
                        meth( env )
            self.runbegun = False
            
        elif svc == xtc.TransitionId.BeginCalibCycle :
            
            for userana in self.userObjects :
                if hasattr(userana, 'begincalibcycle'):
                    userana.begincalibcycle( evt, env )
            self.calibbegun = True
            
        elif svc == xtc.TransitionId.EndCalibCycle :
            
            for userana in self.userObjects :
                if hasattr(userana, 'endcalibcycle'):
                    meth = userana.endcalibcycle
                    if meth.func_code.co_argcount == 3:
                        meth( evt, env )
                    else:
                        meth( env )
            self.calibbegun = False
            
        elif svc == xtc.TransitionId.L1Accept :
            
            for userana in self.userObjects :
                
                # When in Skip/Stop state here, will only call the modules
                # which specifically request it
                if evt.status() == pyana.Normal or getattr(userana, 'pyana_get_all_events', False):
                    
                    # call user module
                    code = userana.event( evt, env )
                    
                    if code == pyana.Terminate:
                        # will stop immediately
                        _log.warning("User module %s requested termination", userana.__modname__)
                        evt.setStatus(code)
                        return                    
                    elif code == pyana.Skip:
                        # skip the rest of the modules 
                        _log.debug("User module %s requested event skip", userana.__modname__)
                        if evt.status() == pyana.Normal: evt.setStatus(code)
                    elif code == pyana.Stop:
                        # skip the rest of the modules, even those who want all events
                        _log.info("User module %s requested stop", userana.__modname__)
                        evt.setStatus(code)
                        break
                        


    def finish(self, evt, env):
        
        _log.debug("evt_dispatch.finish: %s", self.__dict__ )
        
        # finish with run first if was not done yet
        if self.calibbegun :
            for userana in self.userObjects : 
                if hasattr(userana, 'endcalibcycle'): 
                    meth = userana.endcalibcycle
                    if meth.func_code.co_argcount == 3:
                        meth( evt, env )
                    else:
                        meth( env )
            self.calibbegun = False

        # finish with run first if was not done yet
        if self.runbegun :
            for userana in self.userObjects : 
                if hasattr(userana, 'endrun'): 
                    meth = userana.endrun
                    if meth.func_code.co_argcount == 3:
                        meth( evt, env )
                    else:
                        meth( env )
            self.runbegun = False

        # run unconfigure if configure was ran before
        if self.lastConfigTime :
            for userana in self.userObjects : 
                meth = userana.endjob
                if meth.func_code.co_argcount == 3:
                    meth( evt, env )
                else:
                    meth( env )
            self.lastConfigTime = None
        