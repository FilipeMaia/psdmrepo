#
# $Id$
#
# Copyright (c) 2010 SLAC National Accelerator Laboratory
# 

"""
This module is a collection classes and methods to create and manage histograms 
from user analysis modules. Current implementation is based on ROOT package 
(http://root.cern.ch/). Histograms objects produced byt this package are actually 
ROOT objects. For detailed description of their interfaces consult ROOT documentation.  
"""


import logging
import os

# only import ROOT when necessary
def _import():
    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.PyConfig.StartGuiThread = False
    from ROOT import TFile
    return ROOT

class HistoMgrRoot(object):
    """
    This class is histogram manager implementation based on ROOT library. Histograms that are 
    created by this manager reside either in memory or in a ROOT file.
    """
    
    def __init__ (self, **kw ):
        """Creates new histogram manager object. Users should not instantiate new objects, 
        instead the environment method hmgr() should be used to obtain existing manager object.

        Keyword arguments: 
        ``file`` - name of the ROOT file to store histograms, if missing then histograms 
        will be memory-resident.
        """

        self._root = _import()

        self.m_file = None
        fname = kw.get('file')
        if fname : self.m_file = self._root.TFile(fname, 'RECREATE')
        
    def h1d(self, *args, **kw):
        """self.h1d(*args, **kw) -> TH1D
        
        Creates 1-dimensional histogram with bin contents stored as double precision numbers.

        Method accepts the same arguments as the constructors of the corresponding C++ ROOT class TH1D. 
        The returned Python object also inherits most of the methods of the C++ class.
        """
        return self._root.TH1D(*args, **kw)

    def h1f(self, *args, **kw):
        """self.h1f(*args, **kw) -> TH1F
        
        Creates 1-dimensional histogram with bin contents stored as single precision numbers.

        Method accepts the same arguments as the constructors of the corresponding C++ ROOT class TH1F. 
        The returned Python object also inherits most of the methods of the C++ class.
        """
        return self._root.TH1F(*args, **kw)

    def h1i(self, *args, **kw):
        """self.h1i(*args, **kw) -> TH1I
        
        Creates 1-dimensional histogram with bin contents stored as integer numbers.

        Method accepts the same arguments as the constructors of the corresponding C++ ROOT class TH1I. 
        The returned Python object also inherits most of the methods of the C++ class.
        """
        return self._root.TH1I(*args, **kw)

    def h2d(self, *args, **kw):
        """self.h2d(*args, **kw) -> TH2D
        
        Creates 2-dimensional histogram with bin contents stored as double precision numbers.

        Method accepts the same arguments as the constructors of the corresponding C++ ROOT class TH2D. 
        The returned Python object also inherits most of the methods of the C++ class.
        """
        return self._root.TH2D(*args, **kw)

    def h2f(self, *args, **kw):
        """self.h2f(*args, **kw) -> TH2F
        
        Creates 2-dimensional histogram with bin contents stored as single precision numbers.

        Method accepts the same arguments as the constructors of the corresponding C++ ROOT class TH2D. 
        The returned Python object also inherits most of the methods of the C++ class.
        """
        return self._root.TH2F(*args, **kw)

    def h2i(self, *args, **kw):
        """self.h2i(*args, **kw) -> TH2I
        
        Creates 2-dimensional histogram with bin contents stored as integer numbers.

        Method accepts the same arguments as the constructors of the corresponding C++ ROOT class TH2I. 
        The returned Python object also inherits most of the methods of the C++ class.
        """
        return self._root.TH2I(*args, **kw)

    def prof(self, *args, **kw):
        """self.prof(*args, **kw) -> TProfile
        
        Creates 1-dimensional profile histogram with bin contents stored as double precision numbers.

        Method accepts the same arguments as the constructors of the corresponding C++ ROOT class TProfile. 
        The returned Python object also inherits most of the methods of the C++ class.
        """
        return self._root.TProfile(*args, **kw)

    def prof2d(self, *args, **kw):
        """self.prof2d(*args, **kw) -> TProfile2D
        
        Creates 2-dimensional profile histogram with bin contents stored as double precision numbers.

        Method accepts the same arguments as the constructors of the corresponding C++ ROOT class TProfile2D. 
        The returned Python object also inherits most of the methods of the C++ class.
        """
        return self._root.TProfile2D(*args, **kw)

    def histos(self):
        """self.histos() -> list
        
        Returns the list of histograms in the current ROOT directory.  User code does not 
        need to use this methods, for framework implementation only.
        """
        return self._root.gDirectory.GetList()

    def file(self):
        """self.file() -> TFile
        
        Returns ROOT file object for this manager, or None if histograms are stored in memory. 
        """
        return self.m_file

    def close(self):
        """self.close()
        
        Write everything to file and close the file.
        """
        if self.m_file :
            self.m_file.Write()
            logging.info("closing ROOT file")
            self.m_file.Close()

_globalHistoMgr = None

def HistoMgr( **kw ):
    """HistoMgr() -> HistoMgrRoot
    
    Factory method which returns an instance of the histogram manager.
    Current implementation always returns the same ROOT manager instance.
    """
    global _globalHistoMgr
    if _globalHistoMgr is None :
        _globalHistoMgr = HistoMgrRoot(**kw)
    return _globalHistoMgr
