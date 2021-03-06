#
# $Id$
#
# Copyright (c) 2010 SLAC National Accelerator Laboratory
# 

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
    
    def __init__ (self, **kw ):

        self._root = _import()

        self.m_file = None
        fname = kw.get('file')
        if fname : self.m_file = self._root.TFile(fname, 'RECREATE')
        
    def h1d(self, *args, **kw):
        return self._root.TH1D(*args, **kw)

    def h1f(self, *args, **kw):
        return self._root.TH1F(*args, **kw)

    def h1i(self, *args, **kw):
        return self._root.TH1I(*args, **kw)

    def h2d(self, *args, **kw):
        return self._root.TH2D(*args, **kw)

    def h2f(self, *args, **kw):
        return self._root.TH2F(*args, **kw)

    def h2i(self, *args, **kw):
        return self._root.TH2I(*args, **kw)

    def prof(self, *args, **kw):
        return self._root.TProfile(*args, **kw)

    def prof2d(self, *args, **kw):
        return self._root.TProfile2D(*args, **kw)

    def histos(self):
        "return the list of all histograms"
        return self._root.gDirectory.GetList()

    def file(self):
        return self.m_file

    def close(self):
        if self.m_file :
            self.m_file.Write()
            logging.info("closing ROOT file")
            self.m_file.Close()

_globalHistoMgr = None

def HistoMgr( **kw ):
    global _globalHistoMgr
    if _globalHistoMgr is None :
        _globalHistoMgr = HistoMgrRoot(**kw)
    return _globalHistoMgr
