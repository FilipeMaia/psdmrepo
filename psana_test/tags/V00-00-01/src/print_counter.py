class print_counter(object):
    def __init__(self):
        pass

    def beginjob(self,evt,env):
        self.run = -1
        
    def beginrun(self,evt,env):
        self.run += 1
        self.calibcycle = -1

    def begincalibcycle(self,evt,env):
        self.calibcycle += 1
        self.eventno = -1

    def event(self,evt,env):
        self.eventno += 1
        print "run=", self.run, " calibcycle=", self.calibcycle," event=",self.event
