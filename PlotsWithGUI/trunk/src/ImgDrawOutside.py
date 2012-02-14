#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ImgDrawOutside...
#
#------------------------------------------------------------------------

"""Additional graphics outside of 2-d image

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule
@version $Id: 
@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
#import os

import ImgDrawSpectrum  as sp                                   # <======= Depends on form etc.
import ImgDrawProfile   as pr                                   # <======= Depends on form etc.
import ImgDrawZoom      as zo                                   # <======= Depends on form etc.
import ImgDrawProjXY    as xy                                   # <======= Depends on form etc.

#---------------------
#  Class definition --
#---------------------

class ImgDrawOutside :
    """Additional graphics outside of the 2-d image"""

    def __init__(self, icp=None):
        self.icp          = icp
        self.icp.idrawout = self
        self.spec = sp.ImgDrawSpectrum(icp)                     # <======= Depends on form etc.
        self.prof = pr.ImgDrawProfile (icp)                     # <======= Depends on form etc.
        self.zoom = zo.ImgDrawZoom    (icp)                     # <======= Depends on form etc.
        self.prxy = xy.ImgDrawProjXY  (icp)                     # <======= Depends on form etc.


    def get_control(self) :
        return self.icp.control

#-----------------------------

    def draw_outside(self) :
        self.spec.draw_outside_plots_for_list_of_objs(self.arr) # <======= Depends on form etc.
        self.prof.draw_outside_plots_for_list_of_objs(self.arr) # <======= Depends on form etc.
        self.zoom.draw_outside_plots_for_list_of_objs(self.arr) # <======= Depends on form etc.
        self.prxy.draw_outside_plots_for_list_of_objs(self.arr) # <======= Depends on form etc.

    def remove_outside_plot_for_obj(self, obj) :

        try :
            number = obj.get_fig_outside().number
            self.get_control().signal_and_close_fig(number)
        except :
            print 'ImgDrawOutside : remove_outside_plot_for_obj() : WARNING! try to remove non-existent figure...'

        



#-----------------------------
# Test
#-----------------------------

def main():
    w = ImgDrawOutside()

#-----------------------------

if __name__ == "__main__" :
    #main()
    sys.exit ('Module is not supposed to be run for test...')

#-----------------------------
