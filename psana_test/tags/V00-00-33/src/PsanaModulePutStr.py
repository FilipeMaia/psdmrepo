#######################################
# testing module that puts a string with key 'key'
# into the event store during beginJob and event
#
# To be used with testing module PsanaModuleGetStr
# 
class PsanaModulePutStr(object):
    def beginjob(self,evt,env):
        evt.put("This is a string","key")
    def event(self,evt,env):
        evt.put("This is a string","key")

    
