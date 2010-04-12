#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module merger...
#
#------------------------------------------------------------------------

"""Code that merges results of processing from individual jobs 

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
import shutil
import numpy as np

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
import ROOT

#----------------------------------
# Local non-exported definitions --
#----------------------------------
_log = logging.getLogger('pyana.merger')

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class merger ( object ) :

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self ) :
        
        self._files = {}
        self._histos = {}

    #-------------------
    #  Public methods --
    #-------------------

    def merge( self, result ) :

        _log.info('merging data: %s', result)
        if not result: return
        
        files = result.get("files",{})
        self.__mergeFiles( files )

        histos = result.get("histos",[])
        self.__mergeHistos( histos )

    def finish(self, env):
        
        # close all files
        for file in self._files.itervalues() : file.close()

        # store histograms
        env.hmgr().file().cd()
        for h in self._histos.itervalues(): 
            h.Write()

    #--------------------------------
    #  Static/class public methods --
    #--------------------------------

    #--------------------
    #  Private methods --
    #--------------------

    def __mergeFiles(self, files):
        
        for name, tmpname in files.iteritems() :

            try :
                
                _log.debug('merging file %s -> %s', tmpname, name)
                
                # open output file if needed
                file = self._files.get(name)
                if not file :
                    file = open(name, 'wb')
                    self._files[name] = file
            
                # open input file
                tmpfile = open(tmpname)
                
                # copy data in chunks of 1MB
                shutil.copyfileobj(tmpfile, file, 1024*1024)
                
                # delete the temporary file, but do not stop if it fails
                try :
                    os.remove(tmpname)
                except StandardError, exc:
                    _log.error('failed to remove file after merge: %s (%s)', tmpname, exc)

            except Exception, exc:
                
                _log.error('file merge failed for file %s (from %s)', name, tmpname)
                
                
    def __mergeHistos(self, histos):
        
        for h in histos :

            name = h.GetName()

            _log.debug('merging histogram %s', name)

            histo = self._histos.get(name)
            if not histo:
                self._histos[name] = h
            else :
                histo.Add(h)

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
