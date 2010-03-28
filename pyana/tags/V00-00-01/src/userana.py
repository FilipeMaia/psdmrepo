#
# $Id$
#
# Copyright (c) 2010 SLAC National Accelerator Laboratory
# 

import logging

_log = logging.getLogger("pyana.userana")


def mod_import(name):
    """ Helper function to import and instantiate user analysis class """

    # import module
    try :
        mod = __import__(name)
    except ImportError, e :
        _log.exception("Cannot import module %s: %s", name, str(e) )
        return None

    # locate sub-module
    components = name.split('.')
    for comp in components[1:]:
        mod = getattr(mod, comp)
    
    # must define class with the same name
    classname = components[-1]
    userClass = getattr(mod, classname, None)
    if not userClass :
        _log.error("User module %s does not define class %s", name, classname )
        return None
    
    # instantiate it
    try:
        userana = userClass()
    except Exception, e :
        _log.exception("Failure while instantiating class %s: %s", classname, str(e) )
        return None

    # some weird cases are possible, just make sure that callable
    # returns an object which has a list of attributes that we need
    if not userana :
        _log.error("Failure while calling %s(): None returned", classname, str(e) )
        return None

    for method in ['beginjob', 'beginrun', 'event', 'endrun', 'endjob' ] :
        if not hasattr(userana, method) :
            _log.error("User analysis class %s does not define method %s", classname, method )
            return None

    # looks OK so far
    return userana


        