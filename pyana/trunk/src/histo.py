#
# $Id$
#
# Copyright (c) 2010 SLAC National Accelerator Laboratory
# 

import logging
import os

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
from ROOT import TFile


class HistoMgrRoot(object):
    
    def __init__ (self, **kw ):
    
        self.m_file = None
        fname = kw.get('file')
        if fname : self.m_file = TFile(fname, 'RECREATE')
        
    def h1d(self, *args, **kw):
        return ROOT.TH1D(*args, **kw)

    def h1f(self, *args, **kw):
        return ROOT.TH1F(*args, **kw)

    def h2d(self, *args, **kw):
        return ROOT.TH2D(*args, **kw)

    def h2f(self, *args, **kw):
        return ROOT.TH2F(*args, **kw)

    def prof(self, *args, **kw):
        return ROOT.TProfile(*args, **kw)

    def prof2d(self, *args, **kw):
        return ROOT.TProfile2D(*args, **kw)

    def histos(self):
        "return the list of all histograms"
        return ROOT.gDirectory.GetList()

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
