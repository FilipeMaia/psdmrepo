#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Pyana user analysis module image_save_in_file...
#
#------------------------------------------------------------------------

"""User analysis module 

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id: template!pyana-module!py 2987 2012-02-25 03:28:58Z salnikov@SLAC.STANFORD.EDU $

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision: 2987 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import logging

#-----------------------------
# Imports for other modules --
#-----------------------------
#from pypdsdata import xtc

from pypdsdata.xtc import *
from psana import *
import numpy as np

class image_save_in_file (object) :
    """Saves image array in file with specified in the name type."""

    def __init__ ( self ) :
        """Class constructor.
        Parameters are passed from pyana.cfg configuration file.
        All parameters are passed as strings

        @param key_in      string, keyword for input amge array of variable shape
        @param ofname      string, output file name (type is selected by extention)
        @param print_bits  int, bit-word for verbosity control 
        """

        self.m_key_in     = self.configStr  ('key_in',    'image')
        self.m_ofname     = self.configStr  ('ofname',    'img-cspad')
        self.m_print_bits = self.configInt  ('print_bits', 1)

        if self.m_print_bits & 1 : self.print_input_pars()


    def beginjob( self, evt, env ) :
        """This method is called once at the beginning of the job. It should
        do a one-time initialization possible extracting values from event
        data (which is a Configure object) or environment.

        @param evt    event data object
        @param env    environment object
        """
        logging.info( "image_save_in_file.beginjob() called" )

 
    def beginrun( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new run.

        @param evt    event data object
        @param env    environment object
        """
        logging.info( "image_save_in_file.beginrun() called" )

        self.run   = evt.run()
        self.evnum = 0


    def event( self, evt, env ) :
        """This method is called for every L1Accept transition.

        @param evt    event data object
        @param env    environment object
        """
        try :
            self.image = evt.get(self.m_key_in)

        except :
            return

        self.evnum += 1

        if self.m_print_bits & 8 : self.print_part_of_image_array()

        fname = '%s-%04d-%06d.txt' % (self.m_ofname, self.run, self.evnum)
        if self.m_print_bits & 8 : print 'Save image in file =', fname

        np.savetxt(fname, self.image, fmt='%f')


    def endjob( self, evt, env ) :
        """This method is called at the end of the job. It should do 
        final cleanup, e.g. close all open files.
        
        @param evt    event data object
        @param env    environment object
        """        
        logging.info( "image_save_in_file.endjob() called" )


    def print_input_pars( self ) :
        msg = '\nList of input parameters\n  key_in %s\n  ofname %s\n  print_bits: %4d' % \
              (self.m_key_in, self.m_ofname, self.m_print_bits)
        #logging.info( msg )
        print msg


    def print_part_of_image_array( self, r1=50, r2=60, c1=100, c2=110 ) :
        print 'image[%d:%d,%d:%d]:\n' % (r1, r2, c1, c2)
        print self.image[r1:r2,c1:c2]
        print 'image.shape =', self.image.shape

#-----------------------------
#-----------------------------
#-----------------------------
