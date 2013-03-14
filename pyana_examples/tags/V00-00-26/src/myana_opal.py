#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: myana_opal.py 1253 2010-10-14 22:27:28Z salnikov $
#
# Description:
#  Module myana...
#
#------------------------------------------------------------------------

"""User analysis job for XTC data.

This software was developed for the LUSI project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: myana_opal.py 1253 2010-10-14 22:27:28Z salnikov $

@author Andrei Salnikov
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 1253 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import logging

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
from pypdsdata.xtc import *

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class myana_opal ( object ) :
    """Example analysis module which dumps ControlConfig object info."""

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, address = None ) :
        """Constructor. """
        self.address = address

    #-------------------
    #  Public methods --
    #-------------------

    def beginjob( self, evt, env ) :

        # get configuration object
        logging.debug("Get Opal1k config from address %s", self.address)
        config = env.getOpal1kConfig(self.address)
        if not config:
            print '*** opal config object is missing ***'
            return
        
        # dumpt complete configuration
        print "Opal1k config: size=%d" % config.size()
        print "  black_level:", config.black_level()
        print "  gain_percent:", config.gain_percent()
        print "  output_offset:", config.output_offset()
        print "  output_resolution:", str(config.output_resolution())
        print "  output_resolution_bits:", config.output_resolution_bits()
        print "  vertical_binning:", str(config.vertical_binning())
        print "  output_mirroring:", str(config.output_mirroring())
        print "  vertical_remapping:", config.vertical_remapping()
        print "  defect_pixel_correction_enabled:", config.defect_pixel_correction_enabled()
        print "  output_lookup_table_enabled:", config.output_lookup_table_enabled()
        print "  number_of_defect_pixels:", config.number_of_defect_pixels()
        print "  output_lookup_table:", config.output_lookup_table()
        print "  defect_pixel_coordinates:", config.defect_pixel_coordinates()


    def event( self, evt, env ) :

        # Try to get a frame from event. There are multiple devices 
        # that produce Frame type so one has to use address to 
        # chose one specific device. Use config file to change it.
        logging.debug("Get Opal1k frame from address %s", self.address)
        opal = evt.getOpal1kValue(self.address)
        if not opal:
            print '*** opal information is missing ***'
            return

        print "Opal1k frame %dx%dx%d" % (opal.width(), opal.height(), opal.depth())
        
        data = opal.data()
        print "  data shape =", data.shape

    
    def endjob( self, env ) :
        pass
