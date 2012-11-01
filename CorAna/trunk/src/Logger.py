#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Logger...
#
#------------------------------------------------------------------------

"""Is intended as a log-book for this project

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id: template!python!py 4 2008-10-08 19:27:36Z salnikov $

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os
from time import localtime, strftime

#-----------------------------

class Logger :
    """Is intended as a log-book for this project.
    """

    def __init__ ( self, fname=None, level='info' ) :
        """Constructor.
        @param fname  the file name for output log file
        """
        self.fname = fname
        self.level_thr = level
        self.levels=['debug','info','warning','error','crytical']
        self.level_thr_ind = self.levels.index(level)

        self.log_txt = ''
        self.startLog()

    def debug( self, msg, name=None ) :
        self.message(msg, self.levels[0], name)

    def info( self, msg, name=None ) :
        self.message(msg, self.levels[1], name)

    def warning( self, msg, name=None ) :
        self.message(msg, self.levels[2], name)

    def error( self, msg, name=None ) :
        self.message(msg, self.levels[3], name)

    def crytical( self, msg, name=None ) :
        self.message(msg, self.levels[4], name)

    def message( self, msg, level, name=None ) :
        if(self.levels.index(level) < self.level_thr_ind) : return

        self.msg_tot = '' 
        if name is not None :
            self.msg_tot  = self.time_stamp()
            self.msg_tot += ' (' + level + ') '
            self.msg_tot += name + ': '
        else :
            self.msg_tot += '::::'
        self.msg_tot += msg

        self.log_txt += self.msg_tot + '\n'       

        self.appendGUILog(self.msg_tot)
        #print self.msg_tot

    def appendGUILog(self, msg='') :
        try    : self.guilogger.appendGUILog(msg)
        except : pass

    def setGUILogger(self, guilogger) :
        self.guilogger = guilogger

    def time_stamp( self, fmt='%Y-%m-%d %H:%M:%S' ) : # '%Y-%m-%d %H:%M:%S %Z'
        return strftime(fmt, localtime())

    def startLog(self) :
        self.str_start_time = self.time_stamp( fmt='%Y-%m-%d-%H%M%S' )
        if  self.fname == None :
            self.fname = self.str_start_time + '-log.txt'
        self.info('startLog: Start log for file: '+self.fname, __name__)

    def getLogFileName(self):
        return self.fname

    def getLogContent(self):
        return self.log_txt

    def saveLogInFile(self, fname=None):
        if fname == None :
            self.fname_log = self.fname
        else :
            self.fname_log = fname

        self.info('saveLogInFile: '+self.fname_log, __name__)
        f=open(self.fname_log,'w')
        f.write(self.log_txt)
        f.close() 

#-----------------------------

logger = Logger (fname=None)

#-----------------------------
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    logger.debug   ('This is a test message 1')
    logger.info    ('This is a test message 2')
    logger.warning ('This is a test message 3', __name__)
    logger.error   ('This is a test message 4', __name__)
    logger.crytical('This is a test message 5', __name__)

    sys.exit ( 'End of test for Logger' )

#-----------------------------
