import psana
from psana_test import obj2str, epicsPvToStr
from psana_test import types_to_str
import numpy as np

class dump(object):
    def __init__(self):
        self.epicsParam = self.configBool('epics', True)
        self.aliasesParam  = self.configBool('aliases', True)
        if not self.epicsParam:
            self.aliasesParam = False
        self.configParam = self.configBool('config', True)
        self.counterParam = self.configBool('counter', True)
        # how much to indent sub objects in the printout
        self.indent = self.configInt('indent',2)
        # filter options for the keys found. a list of strings that 
        # we will grep the type/src/key eventKey strings with - first
        # excluding the event key string (if exclude is not empty) and then
        # including the event key string (if include is not empty)
        self.exclude = self.configStr('exclude','').split()
        self.include = self.configStr('include','').split()
        self.previousEpics = {}
        self.previousConfig = {}

    def beginjob(self, evt, env):
        self.runNumber = -1
        print "=================="
        print "=== begin job ===="
        if self.aliasesParam:
            aliases = env.epicsStore().aliases()
            print "Epics Aliases: total = %d" % len(aliases)
            indent = ' '*self.indent
            for alias in aliases:
                print "%s%s" % (indent, alias)
        self.dumpEpics(evt, env)
        self.dumpConfig(evt, env)
        self.dumpEvent(evt, env)

    def beginrun(self, evt, env):
        self.runNumber += 1
        counterStr = ''
        if self.counterParam:
            counterStr = '%d' % self.runNumber
        self.calibNumber = -1
        print "==============================================================="
        print "=== beginrun %s ===" % counterStr
        self.dumpEpics(evt, env)
        self.dumpConfig(evt, env)
        self.dumpEvent(evt, env)

    def begincalibcycle(self, evt, env):
        self.calibNumber += 1
        self.eventNumber = -1
        counterStr = ''
        if self.counterParam:
            counterStr = 'run=%d step=%d' % (self.runNumber, self.calibNumber)
        print "==============================================================="
        print "=== begincalibcycle %s ===" % counterStr
        self.dumpEpics(evt, env)
        self.dumpConfig(evt, env)
        self.dumpEvent(evt, env)

    def event(self,evt,env):
        self.eventNumber += 1
        counterStr = ''
        if self.counterParam:
            counterStr = 'run=%d step=%d event=%d' % (self.runNumber, 
                                                      self.calibNumber, self.eventNumber)
        eventId = evt.get(psana.EventId)
        sec,nsec = eventId.time()
        fid = eventId.fiducials()
        print "==============================================================="
        print "=== event: %s seconds= %s nanoseconds= %s fiducials= %s" % \
            (counterStr, sec, nsec, fid)
        self.dumpEpics(evt, env)
        self.dumpConfig(evt, env)
        self.dumpEvent(evt, env)

    def dumpConfig(self, evt, env):
        if not self.configParam:
            return
        keys = env.configStore().keys()
        for key in keys:
            objStr = getEventObjectStr(evt, env, key, inEvent=False, indent=self.indent)
            if objStr == '':
                continue
            eventKeyStr = getEventKeyStr(key, env.aliasMap())
            if self.filterKey(eventKeyStr):
                continue
            if objStr != self.previousConfig.get(eventKeyStr, ''):
                print eventKeyStr
                print objStr
            self.previousConfig[eventKeyStr]=objStr
            
    def dumpEvent(self, evt, env):
        keys = evt.keys()
        for key in keys:
            objStr = getEventObjectStr(evt, env, key, inEvent=True, indent=self.indent)
            if objStr == '':
                continue
            eventKeyStr = getEventKeyStr(key, env.aliasMap())
            if self.filterKey(eventKeyStr):
                continue
            print eventKeyStr
            print objStr

    def filterKey(self, keyStr):
        if len(self.exclude)>0:
	    for term in self.exclude:
                if keyStr.find(term)>=0:
                    return True
        if len(self.include)>0:
            for term in self.include:
	        if keyStr.find(term)>=0:
                    return False
            return True
        return False
        
    def dumpEpics(self, evt, env):
        if not self.epicsParam:
            return
        epicsStore = env.epicsStore()
        pvNames = epicsStore.pvNames()
        printedEpicsHeader = False
        indent = ' ' * self.indent
        for pvName in pvNames:
            pv = epicsStore.getPV(pvName)
            if not pv:
                continue
            pvStr = epicsPvToStr(pv)
            if pvStr != self.previousEpics.get(pvName, ''):
                if not printedEpicsHeader:
                    printedEpicsHeader = True
                    print "Epics PV"
                print "%spvName=%s %s" % (indent, pvName, pvStr)
            self.previousEpics[pvName]=pvStr

######## helper functions ############
def getEventKeyStr(key, amap=None):
    '''Takes a key from evt.keys()
    '''
    typeStr = str(key).split('type=')[1].split('src=')[0].strip()
    srcStr = str(key.src())
    keyStr = "type=%s src=%s" % (typeStr, srcStr)
    if amap:
        srcAlias = amap.alias(key.src())
        if len(srcAlias)>0:
            keyStr += " alias=%s" % srcAlias
    if key.key():
        keyStr += " key=%s" % key.key()
    return keyStr

def getEventObjectStr(evt, env, key, inEvent, indent=2):
    '''return dump str if xtc type or ndarray. Otherwise an empty string.'''
    tp = key.type()
    if tp is None:
        return ''
    if inEvent:
        obj = evt.get(key.type(), key.src(), key.key())
    else:
        obj = env.configStore().get(key.type(), key.src())
    if obj is None:
        return ''
    try:
        typeid = obj.TypeId
    except AttributeError:
        # check for a numpy array
        if not isinstance(obj, np.ndarray):
            return ''
    return obj2str(obj, indent=indent, lvl=1, methodsep='\n')

