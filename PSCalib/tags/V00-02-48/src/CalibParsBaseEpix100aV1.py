#!/usr/bin/env python
#------------------------------
""":py:class:`CalibParsBaseEpix100aV1` - holds basic calibration metadata parameters for associated detector.

@see :py:class:`PSCalib.CalibPars`, :py:class:`PSCalib.CalibParsStore`.

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Revision: $Revision$

@version $Id$

@author Mikhail S. Dubrovin
"""
#------------------------------

class CalibParsBaseEpix100aV1 :

    ndim = 2 
    rows = 704 
    cols = 768 
    size = rows*cols
    shape = (rows, cols)
    size_cm = 16 
    shape_cm = (size_cm,)
    cmod = (4,1,20,0, 0,0,0,0, 0,0,0,0, 0,0,0,0)
    # 4-Epix100a, 1-median for 16 352x96 banks, 20-maximal allowed correction
         
    def __init__(self) : pass

#------------------------------

