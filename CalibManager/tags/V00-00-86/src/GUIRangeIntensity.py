
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIRangeIntensity...
#
#------------------------------------------------------------------------

"""Intensity range setting GUI

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id$

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports
#--------------------------------

from GUIRange import *

#---------------------
#  Class definition --
#---------------------
class GUIRangeIntensity ( GUIRange ) :
    """Range setting GUI
    @see BaseClass
    @see OtherClass
    """

    def __init__ (self, parent=None, str_from=None, str_to=None, txt_from='valid from', txt_to='to') :
        GUIRange.__init__(self, parent, str_from, str_to, txt_from, txt_to)
        #super(GUIRangeIntensity,self).__init__(parent, str_from, str_to, txt_from, txt_to)

    def setEdiValidators(self):
        #self.edi_from.setValidator(QtGui.QDoubleValidator(-self.vmax, self.vmax, 3, self))
        #self.edi_to  .setValidator(QtGui.QDoubleValidator(-self.vmax, self.vmax, 3, self))
        self.edi_from.setValidator(QtGui.QRegExpValidator(QtCore.QRegExp("[-+]?(\d*[.])?\d+|$"),self))
        self.edi_to  .setValidator(QtGui.QRegExpValidator(QtCore.QRegExp("[-+]?(\d*[.])?\d+|$"),self))


    def showToolTips(self):
        self.edi_from  .setToolTip('Minimal intensity in range.\nAccepts float value or empty field for default.')
        self.edi_to    .setToolTip('Maximal intensity in range.\nAccepts float value or empty field for default.')


    def onEdiFrom(self):
        GUIRange.onEdiFrom(self)
        cp.plot_intens_min.setValue(self.str_from)
        if self.statusButtonsIsGood() : self.redraw()


    def onEdiTo(self):
        GUIRange.onEdiTo(self)
        cp.plot_intens_max.setValue(self.str_to)
        if self.statusButtonsIsGood() : self.redraw()


    def redraw(self) :
        if self.parent is not None : self.parent.widgimage.on_draw()

        
    def setParams(self, str_from=None, str_to=None) :
        self.str_from = str_from if str_from is not None else ''
        self.str_to   = str_to   if str_to   is not None else ''

        pmin = cp.plot_intens_min
        pmax = cp.plot_intens_max

        if pmin.value() != pmin.value_def() : self.str_from = pmin.value()
        if pmax.value() != pmax.value_def() : self.str_to   = pmax.value()


    def setStyle(self):
        GUIRange.setStyle(self)
        #super(GUIRangeIntensity,self).setStyle()
        self.edi_from.setFixedWidth(60)
        self.edi_to  .setFixedWidth(60)


    def statusButtonsIsGood(self):
        if self.str_from == '' and self.str_to == '' : return True
        if self.str_from == '' or  self.str_to == '' :
            msg = '\nBOTH FIELDS MUST BE DEFINED OR EMPTY !!!!!!!!'
            logger.warning(msg, __name__ )            
            return False
        if float(self.str_from) > float(self.str_to) :
            msg  = 'First value in range %s exceeds the second value %s' % (self.str_from, self.str_to)
            msg += '\nRANGE SEQUENCE SHOULD BE FIXED !!!!!!!!'
            logger.warning(msg, __name__ )            
            return False
        return True

#-----------------------------

if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    #ex  = GUIRangeIntensity(None, '-12', '345', 'From:', 'To:')
    ex  = GUIRangeIntensity(None, None, None, 'From:', 'To:')
    ex.move(50,100)
    ex.show()
    app.exec_()

#-----------------------------
