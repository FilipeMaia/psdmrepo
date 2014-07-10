import psana
from psana_test import obj2str, epicsPvToStr
from psana_test import types_to_str
import numpy as np

class dump(object):
    def __init__(self):
        self.epicsParam = self.configBool('epics', True)        
        self.aliasesParam  = self.configBool('aliases', True)
        self.followEpicsAliases = self.configBool('dump_aliases',False)
        self.epicsPrintForRegressionTests = self.configBool('regress_dump',False)
        # set epics_update to 'regress' to update epics for regression tests.
        # In this case, an epics pv will only be printed if the timestamp or dbr type differs
        # and the pvid will not be printed.
        if not self.epicsParam:
            self.aliasesParam = False
            self.followEpicsAliases = False
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
        self.previousEpicsAlias = {}
        self.previousConfig = {}

    def beginjob(self, evt, env):
        self.runNumber = -1
        print "=================="
        print "=== begin job ===="
        if self.aliasesParam:
            aliases = env.epicsStore().aliases()
            print "Epics Aliases: total = %d" % len(aliases)
            indent = ' '*self.indent
            aliases.sort()
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
        toPrint = []
        for key in keys:
            objStr = getEventObjectStr(evt, env, key, inEvent=False, indent=self.indent)
            if objStr == '':
                continue
            eventKeyStr = getEventKeyStr(key, env.aliasMap())
            if self.filterKey(eventKeyStr):
                continue
            if objStr != self.previousConfig.get(eventKeyStr, ''):
                toPrint.append((eventKeyStr,objStr))
            self.previousConfig[eventKeyStr]=objStr
        toPrint.sort()
        for eventKeyStr, objStr in toPrint:
            print eventKeyStr
            print objStr
            
    def dumpEvent(self, evt, env):
        keys = evt.keys()
        toPrint = []
        for key in keys:
            objStr = getEventObjectStr(evt, env, key, inEvent=True, indent=self.indent)
            if objStr == '':
                continue
            eventKeyStr = getEventKeyStr(key, env.aliasMap())
            if self.filterKey(eventKeyStr):
                continue
            toPrint.append((eventKeyStr, objStr))
        toPrint.sort()
        for eventKeyStr, objStr in toPrint:
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
        # dump epics pv's through pv names
        pvNames = epicsStore.pvNames()
        pvNames.sort()
        header = "Epics PV"
        previous = self.previousEpics
        self.dumpEpicsImpl(pvNames, header, previous, epicsStore)
        # dump epics pv's through aliases
        if self.followEpicsAliases:
            aliases = epicsStore.aliases()
            aliases.sort()
            header = "Epics PV Aliases"
            previous = self.previousEpicsAlias
            self.dumpEpicsImpl(aliases, header, previous, epicsStore)

    def dumpEpicsImpl(self, pvNames, header, previous, epicsStore):
        printedEpicsHeader = False
        indent = ' ' * self.indent
        for pvName in pvNames:
            pv = epicsStore.getPV(pvName)
            if not pv:
                continue
            if self.epicsPrintForRegressionTests:
                pvStr = epicsPvToStr(pv,False)
                if pv.isCtrl():
                    pvCmpStr = 'ctrl'
                elif pv.isTime():
                    pvCmpStr = 'sec=%s nsec=%s' % (pv.stamp().sec(), pv.stamp().nsec())
                else:
                    pvCmpStr = pvStr
            else:
                pvStr = epicsPvToStr(pv,True)
                pvCmpStr = pvStr
            if pvCmpStr != previous.get(pvName, ''):
                if not printedEpicsHeader:
                    printedEpicsHeader = True
                    print header
                print "%spvName=%s %s" % (indent, pvName, pvStr)
            previous[pvName]=pvCmpStr
        

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

