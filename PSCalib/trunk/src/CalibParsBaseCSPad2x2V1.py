#!/usr/bin/env python
#------------------------------
""":py:class:`CalibParsBaseCSPad2x2V1` - holds basic calibration metadata parameters for associated detector.

@see :py:class:`PSCalib.CalibPars`, :py:class:`PSCalib.CalibParsStore`.

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Revision: $Revision$

@version $Id$

@author Mikhail S. Dubrovin
"""
#------------------------------

class CalibParsBaseCSPad2x2V1 :

    ndim = 3 
    segs = 2 
    rows = 185 
    cols = 388 
    size = rows*cols*segs; 
    shape = (rows, cols, segs)
    size_cm = 4 
    shape_cm = (size_cm,)
    cmod = (1, 25, 25, 100)
        
    def __init__(self) : pass

#------------------------------

