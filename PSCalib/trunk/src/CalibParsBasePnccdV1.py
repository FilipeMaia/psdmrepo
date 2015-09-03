#!/usr/bin/env python
#------------------------------
""":py:class:`CalibParsBasePnccdV1` - holds basic calibration metadata parameters for associated detector.

@see :py:class:`PSCalib.CalibPars`, :py:class:`PSCalib.CalibParsStore`.

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Revision: $Revision$

@version $Id$

@author Mikhail S. Dubrovin
"""
#------------------------------

class CalibParsBasePnccdV1 :

    ndim = 2 
    rows = 0 # VARIABLE SHAPE DATA PARAMETERS WILL BE TAKEN FROM FILE METADATA
    cols = 0 # VARIABLE SHAPE DATA PARAMETERS WILL BE TAKEN FROM FILE METADATA
    size = rows*cols
    shape = (rows, cols)
    size_cm = 7 
    shape_cm = (size_cm,)
    cmod = (1,50,50,100,1,size,1)
        
    def __init__(self) : pass

#------------------------------

