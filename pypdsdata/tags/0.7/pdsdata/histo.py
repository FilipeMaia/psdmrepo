#
# $Id$
#
# Copyright (c) 2010 SLAC National Accelerator Laboratory
# 

import logging
import os

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
from ROOT import TFile, TProfile


class HistoMgrRoot(object):
    
    def __init__ (self, **kw ):
    
        fname = kw.get('file', "pyana-histo.root")
        self.m_file = TFile(fname, 'RECREATE')
            
    def profile(self, name, title, nbins, xmin, xmax, options=""):
        
        return TProfile(name, title, nbins, xmin, xmax, options)


    def close(self):
        
        if self.m_file : 
            self.m_file.Write()
            self.m_file.Close()

_globalHistoMgr = None

def HistoMgr( **kw ):
    global _globalHistoMgr
    if _globalHistoMgr is None :
        _globalHistoMgr = HistoMgrRoot(**kw)
    return _globalHistoMgr
