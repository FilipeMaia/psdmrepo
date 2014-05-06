import psana
import psana_test.psddl_dump as psddl_dump

class dump(object):
    def __init__(self):
        pass

    def beginjob(self,evt,env):
        print "=================="
        print "=== begin job ===="
        
        keys = env.configStore().keys()
        for key in keys:
            getAndDumpPsddlObject(evt,env,key,False)

    def event(self,evt,env):
        eventId = evt.get(psana.EventId)
        sec,nsec = eventId.time()
        fid = eventId.fiducials()
        print "==============================================================="
        print "=== event: seconds= %s nanoseconds= %s fiducials= %s" % (sec, nsec, fid)
        for key in evt.keys():
            getAndDumpPsddlObject(evt,env,key,True)


INDENT=2

def printEventKey(key):
    typeStr = str(key).split('type=')[1].split('src=')[0].strip()
    srcStr = str(key.src())
    print "type=%s src=%s" % (typeStr, srcStr)

def getAndDumpPsddlObject(evt,env,key,inEvent):
    tp = key.type()
    if tp is None:
        return
    if inEvent:
        obj = evt.get(key.type(), key.src(), key.key())
    else:
        obj = env.configStore().get(key.type(), key.src())
        
    if obj is None:
        return
    try:
        typeid = obj.TypeId
    except AttributeError:
        return
        
    # this is a valid typed object. Let's dump it.
    printEventKey(key)
    INDENT_LEVEL=1
    dumpstr = psddl_dump.obj2str(obj,INDENT,INDENT_LEVEL)
    print dumpstr     
