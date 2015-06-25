class WorkerDataDuplicateTime(Exception):
    def __init__(self, tm):
        super(WorkerDataDuplicateTime, self).__init__("tm %d already stored" % tm)

class WorkerDataNextTimeIsEarliest(Exception):
    def __init__(self, tm):
        super(WorkerDataNextTimeIsEarliest, self).__init__("tm %d is the earliest time" % tm)
