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

@version $Id$

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
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
        self.levels = ['debug','info','warning','error','crytical']
        self.setLevel(level)
        self.selectionIsOn = True # It is used to get total log content
        
        self.log = []
        self.startLog(fname)


    def setLevel(self, level):
        """Sets the threshold level of messages for record selection algorithm"""
        self.level_thr     = level
        self.level_thr_ind = self.levels.index(level)


    def getListOfLevels(self):
        return self.levels


    def getLevel(self):
        return self.level_thr


    def getLogFileName(self):
        return self.fname


    def getLogTotalFileName(self):
        return self.fname_total


    def getStrStartTime(self):
        return self.str_start_time


    def debug   ( self, msg, name=None ) : self._message(msg, 0, name)

    def info    ( self, msg, name=None ) : self._message(msg, 1, name)

    def warning ( self, msg, name=None ) : self._message(msg, 2, name)

    def error   ( self, msg, name=None ) : self._message(msg, 3, name)

    def crytical( self, msg, name=None ) : self._message(msg, 4, name)

    def _message ( self, msg, index, name=None ) :
        """Store input message the 2D tuple of records, send request to append GUI.
        """
        tstamp    = self.timeStamp()
        level     = self.levels[index] 
        rec       = [tstamp, level, index, name, msg]
        self.log.append(rec)

        if self.recordIsSelected( rec ) :         
            str_msg = self.stringForRecord(rec)
            self.appendGUILog(str_msg)
            #print str_msg


    def recordIsSelected( self, rec ):
        """Apply selection algorithms for each record:
           returns True if the record is passed,
                   False - the record is discarded from selected log content.
        """
        if not self.selectionIsOn       : return True
        if rec[2] < self.level_thr_ind  : return False
        else                            : return True


    def stringForRecord( self, rec ):
        """Returns the strind presentation of the log record, which intrinsically is a tuple."""
        tstamp, level, index, name, msg = rec
        self.msg_tot = '' 
        if name is not None :
            self.msg_tot  = tstamp
            self.msg_tot += ' (' + level + ') '
            self.msg_tot += name + ': '
        else :
            self.msg_tot += ': '
        self.msg_tot += msg
        return self.msg_tot


    def appendGUILog(self, msg='') :
        """Append message in GUI, if it is available"""
        try    : self.guilogger.appendGUILog(msg)
        except : pass


    def setGUILogger(self, guilogger) :
        """Receives the reference to GUI"""
        self.guilogger = guilogger


    def timeStamp( self, fmt='%Y-%m-%d %H:%M:%S' ) : # '%Y-%m-%d %H:%M:%S %Z'
        return strftime(fmt, localtime())


    def startLog(self, fname=None) :
        """Logger initialization at start"""
        self.str_start_time = self.timeStamp( fmt='%Y-%m-%d-%H:%M:%S' )
        if  fname is None :
            self.fname       = self.str_start_time + '-log.txt'
            self.fname_total = self.str_start_time + '-log-total.txt'
        else :        
            self.fname       = fname
            self.fname_total = self.fname + '-total' 

        self.info ('Start session log file: ' + self.fname,       __name__)
        self.debug('Total log file name: '    + self.fname_total, __name__)


    def getLogContent(self):
        """Return the text content of the selected log records"""
        self.log_txt = ''
        for rec in self.log :
            if self.recordIsSelected( rec ) :         
                self.log_txt += self.stringForRecord(rec) + '\n'
        return  self.log_txt


    def getLogContentTotal(self):
        """Return the text content of all log records"""
        self.selectionIsOn = False
        log_txt = self.getLogContent()
        self.selectionIsOn = True
        return log_txt


    def saveLogInFile(self, fname=None):
        """Save content of the selected log records in the text file"""
        if fname is None : fname_log = self.fname
        else             : fname_log = fname
        self._saveTextInFile(self.getLogContent(), fname_log)


    def saveLogTotalInFile(self, fname=None):
        """Save content of all log records in the text file"""
        if fname is None : fname_log = self.fname_total
        else             : fname_log = fname
        self._saveTextInFile(self.getLogContentTotal(), fname_log)


    def _saveTextInFile(self, text, fname='log.txt'):
        self.debug('saveTextInFile: ' + fname, __name__)
        f=open(fname,'w')
        f.write(text)
        f.close() 

#-----------------------------

logger = Logger (fname=None)

#-----------------------------
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    #logger.setLevel('debug')
    logger.setLevel('warning')
    
    logger.debug   ('This is a test message 1', __name__)
    logger.info    ('This is a test message 2', __name__)
    logger.warning ('This is a test message 3', __name__)
    logger.error   ('This is a test message 4', __name__)
    logger.crytical('This is a test message 5', __name__)
    logger.crytical('This is a test message 6')

    #logger.saveLogInFile()
    #logger.saveLogTotalInFile()

    print 'getLogContent():\n',      logger.getLogContent()
    print 'getLogContentTotal():\n', logger.getLogContentTotal()

    sys.exit ( 'End of test for Logger' )

#-----------------------------
