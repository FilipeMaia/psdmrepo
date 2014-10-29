#
# Simple queque to manage batch jobs that create missing index files.
#


import time
import os.path 
import sqlite3 as sq

from InterfaceCtlr import LSF


pjoin = os.path.join


class Job:
    """ A single job. A job is the xtc filename and the path """

    def __init__(self, row):
        self.row = row

    @property
    def fn(self):
        return self.row['xtcfn']

    @property
    def path(self):
        return self.row['fpath']
    

class IdxSrvClient:
    """ Used by clients to add request to the queue. Creates a new connection for every request. """

    def __init__(self, db):
        self.db = db

    def add_request(self, xtcfn, path):
        q = JobQueue(self.db)
        rc = q.add_request(xtcfn, path)
        q.conn.close()
        return rc


class JobQueue:
    """ A queue of jobs to create missing index files. 
    Uses sqlite to implement the queue.
    """

    def __init__(self, db):
        self.conn = sq.connect(db, timeout=15)
        self.conn.execute("PRAGMA foreign_keys = on;")
        self.conn.text_factory = str

    def add_request(self, xtcfn, path):
        """Add request. Return -1 if the request already exists."""
        item = (xtcfn, path, int(time.time()))
        cur = self.conn.cursor()

        try:
            cur.execute("INSERT INTO idx (xtcfn,fpath,date_added,status) VALUES (?,?,?,'NEW')", item)
            self.conn.commit()
        except sq.IntegrityError:
            return -1
        return 0

    def alljobs(self):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM idx where status != 'FAILED' and status != 'DONE'")
        rows = cur.fetchall()        
        self.conn.commit()

    def to_submit(self):
        """ get list of jobs that need to be submitted """
        self.conn.row_factory = sq.Row
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM idx where status = 'NEW'")
        rows = cur.fetchall()
        
        for row in rows:
            yield Job(row)

    def active(self):
        self.conn.row_factory = sq.Row
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM idx where status != 'NEW' and status != 'FAIL' and status != 'DONE'")
        rows = cur.fetchall()
        
        for row in rows:
            yield Job(row)

    def job_set_status(self, job, status):
        """ set the status of a job """
        now = int(time.time())
        cur = self.conn.cursor()
        cmd = "update idx set status = '%s', date_status = %d where xtcfn = '%s'" % (
            status, now, job.fn)
        cur.execute(cmd)
        self.conn.commit()
            
    def job_submitted(self, job):
        """ set the status of a job to submitted """
        self.job_set_status(job, 'SUBMIT')

    def job_done(self, job):
        """ set the status of a job to done """
        self.job_set_status(job, 'DONE')


# ========================================================================

class LsfIdxJobs(object):
    """ Submit create index job to batch """

    def __init__(self, release, cmd):
        self.release = release 
        self.cmd = cmd
        
    def run(self,job):
        
        cmd = "cd %s; . $SIT_ROOT/bin/sit_setup.sh; " % self.release 
        cmd = cmd + self.cmd + " " + os.path.join(job.path, job.fn)
        logfile = "/reg/g/psdm/psdatmgr/logs/idx/r%d" % int(time.time())
        bque = "psanaq"
        
        job = LSF.submit(cmd, queue=bque, log=logfile)

    def job_done(self, job):

        index_file = pjoin(job.path, "index", job.fn)
        index_file += ".idx"
        return os.path.exists(index_file)


