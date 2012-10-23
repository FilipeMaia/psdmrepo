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

    def debug( self, msg ) :
        self.message(msg,self.levels[0])

    def info( self, msg ) :
        self.message(msg,self.levels[1])

    def warning( self, msg ) :
        self.message(msg,self.levels[2])

    def error( self, msg ) :
        self.message(msg,self.levels[3])

    def crytical( self, msg ) :
        self.message(msg,self.levels[4])

    def message( self, msg, level ) :
        if(self.levels.index(level) < self.level_thr_ind) : return
        print self.time_stamp() + ' (' + level + ') ' + msg

    def time_stamp( self, fmt='%Y-%m-%d %H:%M:%S' ): # '%Y-%m-%d %H:%M:%S %Z'
        return strftime(fmt, localtime())

#-----------------------------

logger = Logger (fname='log-file.txt')

#-----------------------------
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    logger.debug   ('This is a test message 1')
    logger.info    ('This is a test message 2')
    logger.warning ('This is a test message 3')
    logger.error   ('This is a test message 4')
    logger.crytical('This is a test message 5')

    sys.exit ( 'End of test for Logger' )

#-----------------------------
