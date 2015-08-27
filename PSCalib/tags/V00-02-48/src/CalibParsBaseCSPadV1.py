#!/usr/bin/env python
#------------------------------
""":py:class:`CalibParsBaseCSPadV1` - holds basic calibration metadata parameters for associated detector.

@see :py:class:`PSCalib.CalibPars`, :py:class:`PSCalib.CalibParsStore`.

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Revision: $Revision$

@version $Id$

@author Mikhail S. Dubrovin
"""
#------------------------------

class CalibParsBaseCSPadV1 :

    ndim = 3 
    quads= 4 
    segs = 8 
    rows = 185 
    cols = 388 
    size = quads*segs*rows*cols
    shape = (quads*segs, rows, cols)
    size_cm = 4 
    shape_cm = (size_cm,)
    cmod = (1, 25, 25, 100)
        
    def __init__(self) : pass

#------------------------------

