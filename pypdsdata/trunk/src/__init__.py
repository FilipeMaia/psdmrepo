#
# $Id$
#
# Copyright (c) 2010 SLAC National Accelerator Laboratory
# 

"""
This package is a user-level interface to lower-level :py:mod:`_pdsdata` package. It repeats the 
module structure of :py:mod:`_pdsdata` and defines the same classes but shields user from unnecessary 
details present in :py:mod:`_pdsdata`. In many cases pypdsdata imports the classes from :py:mod:`_pdsdata` without 
any changes, consult :py:mod:`_pdsdata` package documentation for those classes. For other classes (e.g.
:py:class:`pypdsdata.acqiris.DataDescV1`, :py:class:`pypdsdata.pnccd.FrameV1`, etc.) this package provides wrappers which hide complex 
details present in low-level XTC library.

Additionally this package provides :py:mod:`pypdsdata.io` module which facilitates reading and merging 
of the datagrams from multiple input XTC files.
"""

# import stuff from extension module
from _pdsdata import Error

__all__ = ['Error', 'acqiris', 'alias', 'andor', 'bld', 'camera', 'control',  'cspad', 'cspad2x2', 
           'encoder', 'epics', 'evr', 'fccd', 'fli', 'gsc16ai', 'io', 'ipimb', 'l3t', 'lusi', 
           'oceanoptics', 'opal1k', 'pnccd', 'pulnix', 'princeton', 'quartz', 'timepix', 'usdusb',
           'xtc']
