#
# $Id$
#
# Copyright (c) 2010 SLAC National Accelerator Laboratory
# 

"""
This package is a user-level interface to lower-level _pdsdata package. It repeats the 
module structure of _pdsdata and defines the same classes but shields user from unnecessary 
details present in _pdsdata. In many cases pypdsdata imports the classes from _pdsdata without 
any changes, consult _pdsdata package documentation for those classes. For other classes (e.g.
acqiris.DataDescV1, pnccd.FrameV1, etc.) this package provides wrappers which hide complex 
details present in low-level XTC library.

Additionally this package provides 'io' module which facilitates reading and merging 
of the datagrams from multiple input XTC files.
"""

# import stuff from extension module
from _pdsdata import Error

__all__ = ['Error', 'acqiris', 'bld', 'camera', 'control',  'cspad', 'cspad2x2', 'encoder', 
           'epics', 'evr', 'fccd', 'gsc16ai', 'io', 'ipimb', 'lusi', 'opal1k', 
           'pnccd', 'pulnix', 'princeton', 'timepix', 'xtc']
