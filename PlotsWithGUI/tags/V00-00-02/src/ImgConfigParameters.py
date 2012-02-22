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
        self.idrawzoom  = None
        self.idrawprof  = None
        self.idrawprxy  = None
        self.idrawprrp  = None

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
        self.typeProjXY    = 'ProjXY'
        self.typeProjRP    = 'ProjRP'
        self.typeZoom      = 'Zoom'
        self.typeCenter    = 'Center'

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
        self.formWedge    = 'Wedge'
        self.formCenter   = 'Center'
        self.formSector   = 'Sector'
        self.formArc      = 'Arc'

        self.formCurrent  = self.formNone

        #----

        self.imageNext     = 'Next'
        self.imageCurrent  = 'Current'
        self.imagePrevious = 'Previous'
        self.increment     = 1

        #----
        #For main image
        self.gridIsOn      = False
        self.logIsOn       = False

        self.list_of_rects  = []
        self.list_of_lines  = []
        self.list_of_circs  = []
        self.list_of_wedgs  = []
        self.list_of_cents  = []

        #----
        #For Center
        self.x_center       = 250
        self.y_center       = 250
        self.d_center       = 1

        #----
        #For ProjXY image
        self.nx_slices      = 3
        self.ny_slices      = 5

        #----
        #For ProjRP image
        self.n_rings        = 3
        self.n_sects        = 5
        self.r_corr_is_on   = True

        #----

        #For ImgDrawOnTop
        self.lwAdd          = 1   # on top object linewidth for mode Add
        self.colAdd         = 'g' # on top object color for mode Add


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
                nx, ny                 = obj.get_number_of_slices_for_rect()
                i = list_of_objs.index(obj)
                print 'Rect in list: i,t,s,r,x,y,w,h,lw,col,nx,ny = %3d %s %6s %6s %4d %4d %4d %4d %4d %s %4d %4d ' % (i, t, s, r, x, y, w, h, lw, col, nx, ny)


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


        list_of_objs = self.list_of_wedgs
        Nobjs = len(list_of_objs)
        if Nobjs > 0 :
            print 'Number of wedges =', Nobjs
            for obj in list_of_objs :
                (x,y,r,w,t1,t2,lw,col,s,t,rem) = obj.get_list_of_wedge_pars()
                nr, np                 = obj.get_number_of_slices_for_wedge()
                i = list_of_objs.index(obj)
                print 'Rect in list: i,t,s,rem,x,y,r,w,t1,t2,lw,col,nr,np = %3d %s %6s %6s %4d %4d %4d %4d %4d %4d %4d %s %4d %4d ' % (i, t, s, rem, x, y, r, w, t1, t2, lw, col, nr, np)


        list_of_objs = self.list_of_cents
        Nobjs = len(list_of_objs)
        if Nobjs > 0 :
            print 'Number of centers =', Nobjs
            for obj in list_of_objs :
                (xc,yc,xerr,yerr,lw,col,s,t,r) = obj.get_list_of_center_pars()
                i = list_of_objs.index(obj)
                print 'Center in list: i,t,s,r,xc,yc,xe,ye,lw,col = %3d %s %6s %6s %4d %4d %4d %4d %4d %s' % (i, t, s, r, xc, yc, xerr, yerr, lw, col)



        print 'CENT_X_CENTER',   self.x_center
        print 'CENT_Y_CENTER',   self.y_center
        print 'CENT_D_CENTER',   self.d_center

        print 'PRXY_NX_SLICES',  self.nx_slices
        print 'PRXY_NX_SLICES',  self.ny_slices

        print 'PRRP_N_RINGS',    self.n_rings
        print 'PRRP_N_SECTS',    self.n_sects

        print 'SETS_LINE_WIDTH', self.lwAdd
        print 'SETS_COLOR',      self.colAdd   

#---------------------------------------

    def saveImgConfigPars ( self, fname=None ) :
        self.setImgConfigParsFileName(fname)        
        print 'ImgConfigParameters : Save image configuration parameters in the file', self.fname
        space = '    '
        f=open(self.fname,'w')
        f.write('FILE_NAME'            + space + self.fname       + '\n')
        f.write('IMG_NUMBER'           + space + str(self.number) + '\n')


        f.write('CENT_X_CENTER'        + space + str(self.x_center ) + '\n')
        f.write('CENT_Y_CENTER'        + space + str(self.y_center ) + '\n')
        f.write('CENT_D_CENTER'        + space + str(self.d_center ) + '\n')
                                                                   
        f.write('PRXY_NX_SLICES'       + space + str(self.nx_slices) + '\n')
        f.write('PRXY_NX_SLICES'       + space + str(self.ny_slices) + '\n')
                                                                   
        f.write('PRRP_N_RINGS'         + space + str(self.n_rings  ) + '\n')
        f.write('PRRP_N_SECTS'         + space + str(self.n_sects  ) + '\n')
                                                                   
        f.write('SETS_LINE_WIDTH'      + space + str(self.lwAdd    ) + '\n')
        f.write('SETS_COLOR'           + space + str(self.colAdd   ) + '\n')



        self.idrawontop.update_list_of_all_objs()

        list_of_objs = self.list_of_rects
        Nobjs = len(list_of_objs)
        if Nobjs > 0 :
            f.write('\n\nNUMBER_OF_RECTS'  + space + str(Nobjs) + '\n')
            for obj in list_of_objs :
                (x,y,w,h,lw,col,s,t,r) = obj.get_list_of_rect_pars()
                nx, ny                 = obj.get_number_of_slices_for_rect()

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
                f.write(space + 'RECT_NX'          + space + str(nx)  + '\n')
                f.write(space + 'RECT_NY'          + space + str(ny)  + '\n')


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


        list_of_objs = self.list_of_wedgs
        Nobjs = len(list_of_objs)
        if Nobjs > 0 :
            f.write('\n\nNUMBER_OF_WEDGS'  + space + str(Nobjs) + '\n')
            for obj in list_of_objs :
                (x,y,r,w,t1,t2,lw,col,s,t,rem) = obj.get_list_of_wedge_pars()
                nr, np                 = obj.get_number_of_slices_for_wedge()

                i = list_of_objs.index(obj)
                f.write('\n')
                f.write(space + 'WEDG_I'           + space + str(i)   + '\n') # Index
                f.write(space + 'WEDG_T'           + space +     t    + '\n') # Type
                f.write(space + 'WEDG_S'           + space + str(s)   + '\n') # isSelected
                f.write(space + 'WEDG_X'           + space + str(x)   + '\n')
                f.write(space + 'WEDG_Y'           + space + str(y)   + '\n')
                f.write(space + 'WEDG_R'           + space + str(r)   + '\n')
                f.write(space + 'WEDG_W'           + space + str(w)   + '\n')
                f.write(space + 'WEDG_T1'          + space + str(t1)  + '\n')
                f.write(space + 'WEDG_T2'          + space + str(t2)  + '\n')
                f.write(space + 'WEDG_L'           + space + str(lw)  + '\n')
                f.write(space + 'WEDG_C'           + space + str(col) + '\n')
                f.write(space + 'WEDG_NR'          + space + str(nr)  + '\n')
                f.write(space + 'WEDG_NP'          + space + str(np)  + '\n')

  
        list_of_objs = self.list_of_cents
        Nobjs = len(list_of_objs)
        if Nobjs > 0 :
            f.write('\n\nNUMBER_OF_CENTS'  + space + str(Nobjs) + '\n')
            for obj in list_of_objs :
                (xc,yc,xerr,yerr,lw,col,s,t,r) = obj.get_list_of_center_pars()
                i = list_of_objs.index(obj)
                f.write('\n')
                f.write(space + 'CENT_I'           + space + str(i)   + '\n') # Index
                f.write(space + 'CENT_T'           + space +     t    + '\n') # Type
                f.write(space + 'CENT_S'           + space + str(s)   + '\n') # isSelected
                f.write(space + 'CENT_X'           + space + str(xc)  + '\n')
                f.write(space + 'CENT_Y'           + space + str(yc)  + '\n')
                f.write(space + 'CENT_XE'          + space + str(xerr)+ '\n')
                f.write(space + 'CENT_YE'          + space + str(yerr)+ '\n')
                f.write(space + 'CENT_L'           + space + str(lw)  + '\n')
                f.write(space + 'CENT_C'           + space + str(col) + '\n')

        f.close() 

#---------------------------------------

    def readImgConfigPars ( self, fname=None ) :
        self.setImgConfigParsFileName(fname)        
        print 'Read parameters from the file ', self.fname
        dicBool = {'false':False, 'true':True}

        self.listOfRectInputParameters = []
        self.listOfLineInputParameters = []
        self.listOfCircInputParameters = []
        self.listOfWedgInputParameters = []
        self.listOfCentInputParameters = []

        if os.path.exists(self.fname) :
            f=open(self.fname,'r')
            for line in f :
                if len(line) == 1 : continue # line is empty
                key = line.split()[0]
                val = line.split()[1]
                if   key == 'FILE_NAME'        : val # self.dirName,self.fileName = os.path.split(val)
                elif key == 'IMG_NUMBER'       : self.number = int(val)

                elif key == 'CENT_X_CENTER'    : self.x_center   = float(val)
                elif key == 'CENT_Y_CENTER'    : self.y_center   = float(val)
                elif key == 'CENT_D_CENTER'    : self.d_center   = float(val)
                                                                
                elif key == 'PRXY_NX_SLICES'   : self.nx_slices  = int(val)
                elif key == 'PRXY_NX_SLICES'   : self.ny_slices  = int(val)
                                                                
                elif key == 'PRRP_N_RINGS'     : self.n_rings    = int(val)
                elif key == 'PRRP_N_SECTS'     : self.n_sects    = int(val)
                                                                
                elif key == 'SETS_LINE_WIDTH'  : self.lwAdd      = int(val)
                elif key == 'SETS_COLOR'       : self.colAdd     = str(val)


                elif key == 'NUMBER_OF_RECTS'  : self.Nrects = int(val)
                elif key == 'RECT_I'           :
                    self.ind    = int(val)
                    self.listOfRectInputParameters.append( [self.typeNone, False, 100, 200, 300, 400, 2, 'r', 1, 1] )
                elif key == 'RECT_T'           : self.listOfRectInputParameters[self.ind][0] = val 
                elif key == 'RECT_S'           : self.listOfRectInputParameters[self.ind][1] = dicBool[val.lower()]
                elif key == 'RECT_X'           : self.listOfRectInputParameters[self.ind][2] = int(val) 
                elif key == 'RECT_Y'           : self.listOfRectInputParameters[self.ind][3] = int(val) 
                elif key == 'RECT_W'           : self.listOfRectInputParameters[self.ind][4] = int(val) 
                elif key == 'RECT_H'           : self.listOfRectInputParameters[self.ind][5] = int(val) 
                elif key == 'RECT_L'           : self.listOfRectInputParameters[self.ind][6] = int(val)
                elif key == 'RECT_C'           : self.listOfRectInputParameters[self.ind][7] = val 
                elif key == 'RECT_NX'          : self.listOfRectInputParameters[self.ind][8] = int(val) 
                elif key == 'RECT_NY'          : self.listOfRectInputParameters[self.ind][9] = int(val) 


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


                elif key == 'NUMBER_OF_WEDGS'  : self.Nwedgs = int(val)
                elif key == 'WEDG_I'           :
                    self.ind    = int(val)
                    self.listOfWedgInputParameters.append( [self.typeNone, False, 100, 200, 300, 50, 0, 30, 2, 'r', 1, 1] )
                elif key == 'WEDG_T'           : self.listOfWedgInputParameters[self.ind][ 0] = val 
                elif key == 'WEDG_S'           : self.listOfWedgInputParameters[self.ind][ 1] = dicBool[val.lower()]
                elif key == 'WEDG_X'           : self.listOfWedgInputParameters[self.ind][ 2] = float(val) 
                elif key == 'WEDG_Y'           : self.listOfWedgInputParameters[self.ind][ 3] = float(val) 
                elif key == 'WEDG_R'           : self.listOfWedgInputParameters[self.ind][ 4] = float(val) 
                elif key == 'WEDG_W'           : self.listOfWedgInputParameters[self.ind][ 5] = float(val) 
                elif key == 'WEDG_T1'          : self.listOfWedgInputParameters[self.ind][ 6] = float(val) 
                elif key == 'WEDG_T2'          : self.listOfWedgInputParameters[self.ind][ 7] = float(val) 
                elif key == 'WEDG_L'           : self.listOfWedgInputParameters[self.ind][ 8] = int(val)
                elif key == 'WEDG_C'           : self.listOfWedgInputParameters[self.ind][ 9] = val 
                elif key == 'WEDG_NR'          : self.listOfWedgInputParameters[self.ind][10] = int(val) 
                elif key == 'WEDG_NP'          : self.listOfWedgInputParameters[self.ind][11] = int(val) 


                elif key == 'NUMBER_OF_CENTS'  : self.Ncents = int(val)
                elif key == 'CENT_I'           :
                    self.ind    = int(val)
                    self.listOfCentInputParameters.append( [self.typeNone, False, 100, 200, 10, 20, 2, 'r'] )
                elif key == 'CENT_T'           : self.listOfCentInputParameters[self.ind][0] = val 
                elif key == 'CENT_S'           : self.listOfCentInputParameters[self.ind][1] = dicBool[val.lower()]
                elif key == 'CENT_X'           : self.listOfCentInputParameters[self.ind][2] = int(val) 
                elif key == 'CENT_Y'           : self.listOfCentInputParameters[self.ind][3] = int(val) 
                elif key == 'CENT_XE'          : self.listOfCentInputParameters[self.ind][4] = int(val) 
                elif key == 'CENT_YE'          : self.listOfCentInputParameters[self.ind][5] = int(val) 
                elif key == 'CENT_L'           : self.listOfCentInputParameters[self.ind][6] = int(val)
                elif key == 'CENT_C'           : self.listOfCentInputParameters[self.ind][7] = val 


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
