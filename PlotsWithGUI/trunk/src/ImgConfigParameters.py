#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ConfigParameters...
#
#------------------------------------------------------------------------

"""Configuration parameters for Img.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id:$

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
import os

#---------------------
#  Class definition --
#---------------------
class ImgConfigParameters :
    """This class contains the configuration parameters of a single image object
    """

    def __init__ ( self, number ) :

        self.number     = number
        self.icpfname   = 'confpars.%03d' % number

        self.control    = None
        self.wimg       = None
        self.wgui       = None
        self.idrawontop = None
        self.idrawout   = None
        self.idrawspec  = None
        self.idrawprof  = None

        self.setRunTimeParametersInit()
        self.setDefaultParameters()
        self.readImgConfigPars()

        #print 'ImgConfigParameters initialization'

#---------------------------------------

    def print_icp( self ) :
        print 'ImgConfigParameters : print for object'

#---------------------------------------

    def setRunTimeParametersInit ( self ) :
        #self.ImgGUIIsOpen         = False
        #self.nWindows            = 3
        #self.widg_img             = None # widget of the image for access
        pass

#---------------------------------------

    def setDefaultParameters ( self ) :
        """Set default configuration parameters hardwired in this module"""
        #print 'setDefaultParameters'
        #----
        #Type is set in the ImgGUI sub-menu
        self.typeNone      = 'None'
        self.typeSpectrum  = 'Spectrum'
        self.typeProfile   = 'Profile'
        self.typeProjX     = 'ProjX'
        self.typeProjY     = 'ProjY'
        self.typeProjR     = 'ProjR'
        self.typeProjP     = 'ProjP'
        self.typeZoom      = 'Zoom'

        self.typeCurrent   = self.typeNone

        #----
        #Mode is set in the ImgControl in the signal methods
        self.modeNone      = 'None'
        self.modeMove      = 'Move'
        self.modeAdd       = 'Add'
        self.modeSelect    = 'Select'
        self.modeOverlay   = 'Overlay'
        self.modeRemove    = 'Remove'

        self.modeCurrent   = self.modeMove

        #----
        #Form is set in the ImgControl in the signal methods
        self.formNone     = 'None'
        self.formRect     = 'Rect'
        self.formLine     = 'Line'
        self.formCircle   = 'Circle'
        self.formSector   = 'Sector'
        self.formArc      = 'Arc'
        self.formCenter   = 'Center'

        self.formCurrent  = self.formNone

        #----
        #For main image
        self.gridIsOn      = False
        self.logIsOn       = False

        self.list_of_rects  = []
        self.list_of_lines  = []
        self.list_of_circs  = []

#---------------------------------------

    def getValIntOrNone( self, val ) :
        if val == 'None' : return None
        else :             return int(val)

#---------------------------------------

    def setImgConfigParsFileName(self, fname=None) :
        if fname == None :
            self.fname = self.icpfname
        else :
            self.fname = fname

#---------------------------------------

    def printImgConfigPars ( self ) :
        """Prints current values of configuration parameters
        """
        print '\nImgConfigParameters'
        print 'File name :', self.icpfname 
        print 'Image No. :', self.number 


        self.idrawontop.update_list_of_all_objs()
        #self.idrawontop.update_list_of_objs( self.list_of_rects ) 

        list_of_objs = self.list_of_rects
        Nobjs = len(list_of_objs)
        if Nobjs > 0 :
            print 'Number of rects =', Nobjs
            for obj in list_of_objs :
                (x,y,w,h,lw,col,s,t,r) = obj.get_list_of_rect_pars()
                i = list_of_objs.index(obj)
                print 'Rect in list: i,t,s,r,x,y,w,h,lw,col = %3d %s %6s %6s %4d %4d %4d %4d %4d %s' % (i, t, s, r, x, y, w, h, lw, col)


        list_of_objs = self.list_of_lines
        Nobjs = len(list_of_objs)
        if Nobjs > 0 :
            print 'Number of lines =', Nobjs
            for obj in list_of_objs :
                (x1,x2,y1,y2,lw,col,s,t,r) = obj.get_list_of_line_pars()
                i = list_of_objs.index(obj)
                print 'Line in list: i,t,s,r,x1,x2,y1,y2,lw,col = %3d %s %6s %6s %4d %4d %4d %4d %4d %s' % (i, t, s, r, x1, x2, y1, y2, lw, col)


        list_of_objs = self.list_of_circs
        Nobjs = len(list_of_objs)
        if Nobjs > 0 :
            print 'Number of circs =', Nobjs
            for obj in list_of_objs :
                (x,y,r0,lw,col,s,t,r) = obj.get_list_of_rect_pars()
                i = list_of_objs.index(obj)
                print 'Circ in list: i,t,s,r,x,y,r,lw,col = %3d %s %6s %6s %4d %4d %4d %4d %s' % (i, t, s, r, x, y, r0, lw, col)


#---------------------------------------

    def saveImgConfigPars ( self, fname=None ) :
        self.setImgConfigParsFileName(fname)        
        print 'ImgConfigParameters : Save image configuration parameters in the file', self.fname
        space = '    '
        f=open(self.fname,'w')
        f.write('FILE_NAME'            + space + self.fname       + '\n')
        f.write('IMG_NUMBER'           + space + str(self.number) + '\n')

        self.idrawontop.update_list_of_all_objs()

        list_of_objs = self.list_of_rects
        Nobjs = len(list_of_objs)
        if Nobjs > 0 :
            f.write('\n\nNUMBER_OF_RECTS'  + space + str(Nobjs) + '\n')
            for obj in list_of_objs :
                (x,y,w,h,lw,col,s,t,r) = obj.get_list_of_rect_pars()
                i = list_of_objs.index(obj)
                f.write('\n')
                f.write(space + 'RECT_I'           + space + str(i)   + '\n') # Index
                f.write(space + 'RECT_T'           + space +     t    + '\n') # Type
                f.write(space + 'RECT_S'           + space + str(s)   + '\n') # isSelected
                f.write(space + 'RECT_X'           + space + str(x)   + '\n')
                f.write(space + 'RECT_Y'           + space + str(y)   + '\n')
                f.write(space + 'RECT_W'           + space + str(w)   + '\n')
                f.write(space + 'RECT_H'           + space + str(h)   + '\n')
                f.write(space + 'RECT_L'           + space + str(lw)  + '\n')
                f.write(space + 'RECT_C'           + space + str(col) + '\n')


        list_of_objs = self.list_of_lines
        Nobjs = len(list_of_objs)
        if Nobjs > 0 :
            f.write('\n\nNUMBER_OF_LINES'  + space + str(Nobjs) + '\n')
            for obj in list_of_objs :
                (x1,x2,y1,y2,lw,col,s,t,r) = obj.get_list_of_line_pars()
                i = list_of_objs.index(obj)
                f.write('\n')
                f.write(space + 'LINE_I'           + space + str(i)   + '\n') # Index
                f.write(space + 'LINE_T'           + space +     t    + '\n') # Type
                f.write(space + 'LINE_S'           + space + str(s)   + '\n') # isSelected
                f.write(space + 'LINE_X1'          + space + str(x1)  + '\n')
                f.write(space + 'LINE_X2'          + space + str(x2)  + '\n')
                f.write(space + 'LINE_Y1'          + space + str(y1)  + '\n')
                f.write(space + 'LINE_Y2'          + space + str(y2)  + '\n')
                f.write(space + 'LINE_L'           + space + str(lw)  + '\n')
                f.write(space + 'LINE_C'           + space + str(col) + '\n')


        list_of_objs = self.list_of_circs
        Nobjs = len(list_of_objs)
        if Nobjs > 0 :
            f.write('\n\nNUMBER_OF_CIRCS'  + space + str(Nobjs) + '\n')
            for obj in list_of_objs :
                (x,y,r,lw,col,s,t,r) = obj.get_list_of_circ_pars()
                i = list_of_objs.index(obj)
                f.write('\n')
                f.write(space + 'CIRC_I'           + space + str(i)   + '\n') # Index
                f.write(space + 'CIRC_T'           + space +     t    + '\n') # Type
                f.write(space + 'CIRC_S'           + space + str(s)   + '\n') # isSelected
                f.write(space + 'CIRC_X'           + space + str(x)   + '\n')
                f.write(space + 'CIRC_Y'           + space + str(y)   + '\n')
                f.write(space + 'CIRC_R'           + space + str(r)   + '\n')
                f.write(space + 'CIRC_L'           + space + str(lw)  + '\n')
                f.write(space + 'CIRC_C'           + space + str(col) + '\n')

        f.close() 

#---------------------------------------

    def readImgConfigPars ( self, fname=None ) :
        self.setImgConfigParsFileName(fname)        
        print 'Read parameters from the file ', self.fname
        dicBool = {'false':False, 'true':True}

        self.listOfRectInputParameters = []
        self.listOfLineInputParameters = []
        self.listOfCircInputParameters = []

        if os.path.exists(self.fname) :
            f=open(self.fname,'r')
            for line in f :
                if len(line) == 1 : continue # line is empty
                key = line.split()[0]
                val = line.split()[1]
                if   key == 'FILE_NAME'        : val # self.dirName,self.fileName = os.path.split(val)
                elif key == 'IMG_NUMBER'       : self.number = int(val)


                elif key == 'NUMBER_OF_RECTS'  : self.Nrects = int(val)
                elif key == 'RECT_I'           :
                    self.ind    = int(val)
                    self.listOfRectInputParameters.append( [self.typeNone, False, 100, 200, 300, 400, 2, 'r'] )
                elif key == 'RECT_T'           : self.listOfRectInputParameters[self.ind][0] = val 
                elif key == 'RECT_S'           : self.listOfRectInputParameters[self.ind][1] = dicBool[val.lower()]
                elif key == 'RECT_X'           : self.listOfRectInputParameters[self.ind][2] = int(val) 
                elif key == 'RECT_Y'           : self.listOfRectInputParameters[self.ind][3] = int(val) 
                elif key == 'RECT_W'           : self.listOfRectInputParameters[self.ind][4] = int(val) 
                elif key == 'RECT_H'           : self.listOfRectInputParameters[self.ind][5] = int(val) 
                elif key == 'RECT_L'           : self.listOfRectInputParameters[self.ind][6] = int(val)
                elif key == 'RECT_C'           : self.listOfRectInputParameters[self.ind][7] = val 


                elif key == 'NUMBER_OF_LINES'  : self.Nlines = int(val)
                elif key == 'LINE_I'           :
                    self.ind    = int(val)
                    self.listOfLineInputParameters.append( [self.typeNone, False, 100, 200, 300, 400, 2, 'r'] )
                elif key == 'LINE_T'           : self.listOfLineInputParameters[self.ind][0] = val 
                elif key == 'LINE_S'           : self.listOfLineInputParameters[self.ind][1] = dicBool[val.lower()]
                elif key == 'LINE_X1'          : self.listOfLineInputParameters[self.ind][2] = int(val) 
                elif key == 'LINE_X2'          : self.listOfLineInputParameters[self.ind][3] = int(val) 
                elif key == 'LINE_Y1'          : self.listOfLineInputParameters[self.ind][4] = int(val) 
                elif key == 'LINE_Y2'          : self.listOfLineInputParameters[self.ind][5] = int(val) 
                elif key == 'LINE_L'           : self.listOfLineInputParameters[self.ind][6] = int(val)
                elif key == 'LINE_C'           : self.listOfLineInputParameters[self.ind][7] = val 


                elif key == 'NUMBER_OF_CIRCS'  : self.Nlines = int(val)
                elif key == 'CIRC_I'           :
                    self.ind    = int(val)
                    self.listOfCircInputParameters.append( [self.typeNone, False, 100, 200, 300, 2, 'r'] )
                elif key == 'CIRC_T'           : self.listOfCircInputParameters[self.ind][0] = val 
                elif key == 'CIRC_S'           : self.listOfCircInputParameters[self.ind][1] = dicBool[val.lower()]
                elif key == 'CIRC_X'           : self.listOfCircInputParameters[self.ind][2] = int(val) 
                elif key == 'CIRC_Y'           : self.listOfCircInputParameters[self.ind][3] = int(val) 
                elif key == 'CIRC_R'           : self.listOfCircInputParameters[self.ind][4] = int(val) 
                elif key == 'CIRC_L'           : self.listOfCircInputParameters[self.ind][5] = int(val)
                elif key == 'CIRC_C'           : self.listOfCircInputParameters[self.ind][6] = val 

            f.close()

        else :
            print 'WEARNING: THE FILE :', self.fname, ' DOES NOT EXIST...'
            return

        for row in self.listOfRectInputParameters :
            print 'rect pars =', row
       
#---------------------------------------
#---------------------------------------
#---------------------------------------

class GlobalImgConfigParameters :
    """This class contains the configuration parameters of all image objects
    """

    def __init__ ( self ) :
        print 'GlobalImgConfigParameters initialization'
        self.dict_img_config_pars = {}


    def addImgConfigPars( self, obj ) :
        objNumber = len(self.dict_img_config_pars) + 1
        print 'Create object No.', objNumber      
        self.dict_img_config_pars[obj] = ImgConfigParameters (objNumber)
        return self.dict_img_config_pars[obj]


    def deleteImgConfigPars( self, obj ) :        
        if obj in self.dict_img_config_pars :
            if len(self.dict_img_config_pars) < 2 :
                self.dict_img_config_pars.clear()
            else :
                del self.dict_img_config_pars[obj]


    def getImgConfigPars( self, obj ) :
        return self.dict_img_config_pars[obj]
 
#---------------------------------------
# Makes a single object of this class --
#---------------------------------------

giconfpars = GlobalImgConfigParameters ()

#---------------------------------------
