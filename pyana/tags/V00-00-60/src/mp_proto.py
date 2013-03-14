#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module mp_proto...
#
#------------------------------------------------------------------------

"""protocol definition for multiprocessing data exchange.

This software was developed for the LUSI project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id$

@author Andrei Salnikov
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
import logging

#---------------------------------
#  Imports of base class module --
#---------------------------------


#-----------------------------
# Imports for other modules --
#-----------------------------
from pypdsdata import xtc
from pypdsdata import epics

#----------------------------------
# Local non-exported definitions --
#----------------------------------


#------------------------
# Exported definitions --
#------------------------

# protocol codes
OP_EVENT = 'evt'       # means next regular event
OP_FINISH = 'fin'      # means end of data
OP_RESULT = 'res'      # means return (partial) result
OP_END = 'end'         # means done, close connection

#---------------------
#  Class definition --
#---------------------
class mp_proto ( object ) :
    """ Class hiding the details of the data exchange """

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, conn, dg_ref, logname ) :
        """ Constructor takes connection object. """

        # define instance variables
        self._conn = conn
        self._dg_ref = dg_ref
        self._log = logging.getLogger(logname)

    #-------------------
    #  Public methods --
    #-------------------

    def close(self):
        return self._conn.close()

    def sendCode ( self, code ) :
        """ Send code only """
        self._conn.send_bytes(code)

    def sendData ( self, data ) :
        """ Send arbitrary data using the underlying connection's send() method """
        self._conn.send(data)

    def sendEventData ( self, dg, fileName, fpos, run, expNum, env ) :
        """ Send event data """

        self.sendCode(OP_EVENT)
        
        mp_proto._sendEpicsList(self._conn, env.m_epics.m_name2epics)
        
        if self._dg_ref :
            self._conn.send(fileName)
            self._conn.send(fpos)
        else :
            self._conn.send_bytes(dg)
        self._conn.send(run)
        self._conn.send(expNum)

    def getRequest(self):
        """ Generator function that returns all requests """

        files = {}
        
        while True :
        
            opcode = self._conn.recv_bytes()
            
            if opcode == OP_END : 
                
                yield (opcode,)
    
            elif opcode == OP_FINISH : 
                
                yield (opcode,)
    
            elif opcode == OP_EVENT :
                
                try :
                    
                    # read epics list from pipe
                    epics_data = dict([e for e in mp_proto._getEpics(self._conn)])
                    
                    # read next object from pipe, this would be a datagram sent as a string or 
                    # a file name plus file position
                    if self._dg_ref :
                        
                        fname = self._conn.recv()
                        fpos = self._conn.recv()
                        self._log.debug("fname=%s fpos=%d", fname, fpos )
                        
                        # read datagram from a file
                        file = files.get(fname)
                        if not file :
                            file = open(fname, 'rb')
                            files[fname] = file
                        file.seek(fpos)
                        dgiter = xtc.XtcFileIterator(file)
                        dg = dgiter.next()
                        
                    else :
                        
                        data = self._conn.recv_bytes()
                        self._log.debug("received %s bytes", len(data) )
                        dg = xtc.Dgram(data)

                    run = self._conn.recv()
                    expNum = self._conn.recv()

                    yield (opcode, epics_data, dg, run, expNum)
                    
                except EOFError, ex:
                    
                    self._log.error("server closed connection unexpectedly" )
                    raise

            elif opcode == OP_RESULT :
                
                tag = self._conn.recv()
                yield (opcode, tag)

    def getResult(self, tag=None):
        
        self.sendCode(OP_RESULT)
        self._conn.send(tag)
        return self._conn.recv()
                    
    #--------------------------------
    #  Static/class public methods --
    #--------------------------------

    @staticmethod
    def _sendEpicsList(pipe, epicsList):
        
        for name, epics in epicsList.iteritems() :
            # send as buffer
            pipe.send_bytes(name)
            pipe.send_bytes(epics)
        # EOD
        pipe.send_bytes('$')
    
    # generator for Epics objects read from pipe
    @staticmethod
    def _getEpics(pipe):
    
        while True :
            buf = pipe.recv_bytes()
            if buf == '$' : break
            yield (buf, epics.from_buffer(pipe.recv_bytes()))


    #--------------------
    #  Private methods --
    #--------------------

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
