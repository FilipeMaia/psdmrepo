#
# Simple queque to manage batch jobs that create missing index files.
#

import time
import os.path 
import sqlite3 as sq
from datetime import datetime

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
    
    @property
    def batchid(self):
        return self.row['batchjobid']

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

    def job_set_status(self, job, status, setextra=None):
        """ set the status of a job """
        now = int(time.time())
        cur = self.conn.cursor()
        if setextra:
            cmd = "update idx set status='%s', date_status=%d, %s where xtcfn = '%s'" % (
                status, now, setextra, job.fn)
        else:
            cmd = "update idx set status = '%s', date_status = %d where xtcfn = '%s'" % (
                status, now, job.fn)
        print cmd
        cur.execute(cmd)
        self.conn.commit()
            
    def job_submitted(self, job, jobid):
        """ set the status of a job to submitted """
        self.job_set_status(job, 'SUBMIT', "batchjobid={}".format(jobid))

    def job_done(self, job):
        """ set the status of a job to done """
        self.job_set_status(job, 'DONE', "exitCode=0")

    def job_failed(self, job,rc):
        """ set the status of a job to done """
        self.job_set_status(job, 'FAIL', "exitCode={}".format(rc))
    


# ========================================================================

class LsfIdxJobs(object):
    """ Submit create index job to batch """

    def __init__(self, release, cmd, logdir=''):
        self.release = release 
        self.cmd = cmd
        self.logdir = logdir
        
    def run(self,job):
        
        cmd = ". $SIT_ROOT/bin/sit_setup.sh %s; " % self.release 
        cmd = cmd + self.cmd + " " + os.path.join(job.path, job.fn)
        logfile = "{}/d{}.log".format(self.logdir, datetime.now().strftime("%Y%m%dT%H%M%S")) 
        bque = "psanaq"

        job = LSF.submit(cmd, queue=bque, log=logfile)
        return job


    def job_done(self, job):

        index_file = pjoin(job.path, "index", job.fn)
        index_file += ".idx"
        if os.path.exists(index_file + "erererer"):
            print("index file found {}".format(index_file))
            return True, 0 
        else:
            job = LSF.Job(job.batchid)
            job.update()

            lsf_status = job.status()
            exitCode = job.exitStatus()
            
            print("LSF status {} {}".format(lsf_status, exitCode))
            if lsf_status is None:
                return False, -1

            if lsf_status & (LSF.JOB_STAT_PEND|LSF.JOB_STAT_RUN):
                print("DDD 1")
                return False, -1
            elif lsf_status & (LSF.JOB_STAT_PSUSP|LSF.JOB_STAT_SSUSP|LSF.JOB_STAT_USUSP):
                print("DDD 2")
                return False, -1
            elif lsf_status & LSF.JOB_STAT_UNKWN:
                print("Unknown status")
                return True, 2
            elif lsf_status & LSF.JOB_STAT_DONE:
                print("DDD 3")
                return True, 0
            elif lsf_status & LSF.JOB_STAT_EXIT:
                print("DDD 4")
                return True, exitCode
