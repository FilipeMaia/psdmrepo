import logging


# copied from psana/Module.h
# https://www.slac.stanford.edu/~gapon/TALKS/2013_Oct_LCLS_UserMeeting/Batch_Psana_Intro_v2.pdf pg 7
class event_process(object):
    """
    An empty event process in the new style.
    See some working examples in event_process_lib.py
    They all inherit form this.
    """
    def __init__(self):
        self.output       = event_process_output()
        self.reducer_rank = 0
        self.logger       = logging.getLogger(__name__+'.default_logger')
        return

    def set_parent(self,parent):
        self.parent = parent
        #if 'r{:}'.format(self.parent.rank) not in self.logger.name:
            #self.logger.name = '{:}.r{:}'.format(self.logger.name,self.parent.rank)
        

    def beginJob(self):
        #print "rank {:}".format(self.parent.rank)
        return

    def beginRun(self):
        return

    def event(self, evt):
        return

    def endRun(self):
        return

    def endJob(self):
        return

    def replicate_info(self):
        return None

    def describe_self(self):
        # return a pickleable object (dictionary)
        # that can be used to reproduce this instance (minus the data)
        return (str(self.__class__).split('.')[-1].replace("'>",""), self.replicate_info() )

#    def beginStep(self):
#        return
#    def endStep(self):
#        return

class event_process_output(object):
    """
    structured holder for results of event process.
    """

    def __init__(self):
        self.in_report       = None
        self.in_report_title = None
        self.table           = {}
        self.figures         = {}
        self.text            = []
        return

    def __contains__(self,item):
        if item == 'in_report' or item == 'in_report_title':
            return True
        return self.__dict__[item] and len(self.__dict__[item])>0

    def __getitem__(self,key):
        return self.__dict__[key]

    def __setitem__(self,key,val):
        self.__dict__[key] = val
        return

    #def __repr__(self):
        #out =[]
        #for k in self.__dict__:
            #out.append( "{:} = {:}".format( k, self.__dict__[k] ) )
        #return '\n'.join(out)
