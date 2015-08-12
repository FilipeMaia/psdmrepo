#------------------------------
"""User analysis module 

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id: cspad_gainmap.py 8453 2014-06-20 22:38:14Z cpo@SLAC.STANFORD.EDU $

@author Christopher O'Grady
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision: 8453 $"
# $Source$

import numpy as np
import psana

class cspad_gainmap (object) :
    """Saves image array in file with specified in the name type."""

    def __init__ ( self ) :
        self.gm = np.empty([32,185,388])
        self.m_src = self.configSrc('source')
        self.key_in = self.configStr('key_in')
        self.key_out = self.configStr('key_out')
        self.gain = self.configFloat('gain',6.87526) # from Aaron Brewster

    def beginjob( self, evt, env ) :
        pass
 
    def beginrun( self, evt, env ) :

        configStore = env.configStore()
        self.cfg = configStore.get(psana.CsPad.ConfigV5,self.m_src)
        if self.cfg is None:
            return
        # even running with 1 quad we seem to have 4 quads of config information
        for iquad in range(self.cfg.quads_shape()[0]):
            gm = self.cfg.quads(iquad).gm().gainMap()
            for (row,col), value in np.ndenumerate(gm):
                for i in range(16):
                    iasic = i%2
                    i2x1 = i/2
                    if (gm[row][col] & (1<<i)):
                        # hi gain
                        self.gm[i2x1+iquad*8][row][col+iasic*194]=1
                    else:
                        # low gain
                        self.gm[i2x1+iquad*8][row][col+iasic*194]=self.gain

    def event( self, evt, env ) :
        if self.cfg is None:
            return
        cspad = evt.get(psana.ndarray_float64_3,self.m_src,self.key_in)
        if cspad is None:
            return
        cspad_corr = cspad*self.gm
        cspad = evt.put(cspad_corr,self.m_src,self.key_out)

    def endjob( self, evt, env ) :
        pass
