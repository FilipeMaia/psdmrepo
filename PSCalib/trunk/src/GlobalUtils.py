#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module CalibParsStore...
#
#------------------------------------------------------------------------

"""
:py:class:`PSCalib.GlobalUtils` - contains a set of utilities

Usage::

    # Import
    import PSCalib.GlobalUtils as gu

    # Initialization
    resp = gu.<method(pars)>

@see other interface methods in :py:class:`PSCalib.CalibPars`, :py:class:`PSCalib.CalibParsStore`

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

@version $Id: 2013-03-08$

@author Mikhail S. Dubrovin
"""

#--------------------------------
__version__ = "$Revision$"
#--------------------------------

import numpy as np

#import sys
#------------------------------
#------------------------------

# ATTENTION !!!!! ALL LISTS SHOULD BE IN THE SAME ORDER (FOR DICTIONARIES)

# Enumerated and named parameters

PEDESTALS    = 0
PIXEL_STATUS = 1
PIXEL_RMS    = 2
PIXEL_GAIN   = 3
PIXEL_MASK   = 4
PIXEL_BKGD   = 5
COMMON_MODE  = 6

calib_types  = ( PEDESTALS,   PIXEL_STATUS,   PIXEL_RMS,   PIXEL_GAIN,   PIXEL_MASK,   PIXEL_BKGD,   COMMON_MODE)
calib_names  = ('pedestals', 'pixel_status', 'pixel_rms', 'pixel_gain', 'pixel_mask', 'pixel_bkgd', 'common_mode')
calib_dtypes = ( np.float32,  np.uint16,      np.float32,  np.float32,   np.uint16,    np.float32,   np.double)

dic_calib_type_to_name  = dict(zip(calib_types, calib_names))
dic_calib_name_to_type  = dict(zip(calib_names, calib_types))
dic_calib_type_to_dtype = dict(zip(calib_types, calib_dtypes))

LOADED     = 1
DEFAULT    = 2
UNREADABLE = 3
UNDEFINED  = 4
WRONGSIZE  = 5
NONFOUND   = 6

calib_statvalues = ( LOADED,   DEFAULT,   UNREADABLE,   UNDEFINED,   WRONGSIZE,   NONFOUND)
calib_statnames  = ('LOADED', 'DEFAULT', 'UNREADABLE', 'UNDEFINED', 'WRONGSIZE', 'NONFOUND')

dic_calib_status_value_to_name = dict(zip(calib_statvalues, calib_statnames))
dic_calib_status_name_to_value = dict(zip(calib_statnames,  calib_statvalues))

#------------------------------
#------------------------------
#------------------------------
#------------------------------

UNDEFINED = 0
CSPAD     = 1 
CSPAD2X2  = 2 
PRINCETON = 3 
PNCCD     = 4 
TM6740    = 5 
OPAL1000  = 6 
OPAL2000  = 7 
OPAL4000  = 8 
OPAL8000  = 9 
ORCAFL40  = 10
EPIX      = 11
EPIX10K   = 12
EPIX100A  = 13
FCCD960   = 14
ANDOR     = 15
ACQIRIS   = 16
""" Enumetated detector types"""

list_of_det_type = (UNDEFINED, CSPAD, CSPAD2X2, PRINCETON, PNCCD, TM6740, \
                    OPAL1000, OPAL2000, OPAL4000, OPAL8000, \
                    ORCAFL40, EPIX, EPIX10K, EPIX100A, FCCD960, ANDOR, ACQIRIS)
""" List of enumetated detector types"""

list_of_det_names = ('UNDEFINED', 'CSPAD', 'CSPAD2x2', 'Princeton', 'pnCCD', 'Tm6740', \
                     'Opal1000', 'Opal2000', 'Opal4000', 'Opal8000', \
                     'OrcaFl40', 'Epix', 'Epix10k', 'Epix100a', 'Fccd960', 'Andor', 'Acqiris')
""" List of enumetated detector names"""

list_of_calib_groups = ('UNDEFINED',
                        'CsPad::CalibV1',
                        'CsPad2x2::CalibV1',
                        'Princeton::CalibV1',
                        'PNCCD::CalibV1',
                        'Camera::CalibV1',
                        'Camera::CalibV1',
                        'Camera::CalibV1',
                        'Camera::CalibV1',
                        'Camera::CalibV1',
                        'Camera::CalibV1',
                        'Epix::CalibV1',
                        'Epix10k::CalibV1',
                        'Epix100a::CalibV1',
                        'Camera::CalibV1',
                        'Andor::CalibV1',
                        'Acqiris::CalibV1')
""" List of enumetated detector calibration groups"""

dic_det_type_to_name = dict(zip(list_of_det_type, list_of_det_names))
""" Dictionary for detector type : name"""

dic_det_type_to_calib_group = dict(zip(list_of_det_type, list_of_calib_groups))
""" Dictionary for detector type : group"""

#------------------------------

def det_src_to_type(source) :
    """ Returns enumerated detector type for string source
    """
    if   source.find(':Cspad.')     : return CSPAD
    elif source.find(':Cspad2x2.')  : return CSPAD2X2
    elif source.find(':pnCCD.')     : return PNCCD
    elif source.find(':Princeton.') : return PRINCETON
    elif source.find(':Andor.')     : return ANDOR
    elif source.find(':Epix100a.')  : return EPIX100A
    elif source.find(':Opal1000.')  : return OPAL1000
    elif source.find(':Opal2000.')  : return OPAL2000
    elif source.find(':Opal4000.')  : return OPAL4000
    elif source.find(':Opal8000.')  : return OPAL8000
    elif source.find(':Tm6740.')    : return ORCAFL40
    elif source.find(':OrcaFl40.')  : return ORCAFL40
    elif source.find(':Fccd960.')   : return FCCD960
    elif source.find(':Acqiris.')   : return ACQIRIS
    else                            : return UNDEFINED

#------------------------------
#------------------------------
#------------------------------
#------------------------------
