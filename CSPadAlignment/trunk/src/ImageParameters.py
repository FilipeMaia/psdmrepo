#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ImageParameters...
#
#------------------------------------------------------------------------

"""Module ImageParameters for CSPadAlignment package

CSPadAlignment package is intended to check quality of the CSPad alignment
using image of wires illuminated by flat field.
Shadow of wires are compared with a set of straight lines, which can be
interactively adjusted.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id$

@author Mikhail S. Dubrovin
"""
#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
# $Source$
#------------------------------
#!/usr/bin/env python
#----------------------------------

import os
import sys
import GlobalMethods as gm

#----------------------------------

class ImageParameters :
    """Read from file and hold the image array"""
    def __init__ (self) :

        self.arr=None
        self.setDefaultParameters()
        self.setRunTimeParameters()
        self.readParameters()


    def loadImageArrayFromFile ( self, fname='cspad-ave100-cxi80410-r0628-v02.txt' ) :
        self.arr = gm.getNumpyArrayFromFile(fname)


    def setDefaultParameters ( self ) :
        """Set default configuration parameters hardwired in this module"""
        print 'setDefaultParameters'

        self.AWireMin = 1000
        self.AWireMax = 2200
        self.APeak    = 2652

        self.figNumBase = 10

        self.xmin = xmin = 0
        self.ymin = ymin = 0

        self.xmax = xmax = 850
        self.ymax = ymax = 850

        # For Quad:          Xmin  Xmax  Ymin   Ymax Orient
        self.line_coord = [ [xmin, xmax,  230,  230, 'h'], #0# Horizontal wires
                            [xmin, xmax,  337,  371, 'h'], #1
                            [xmin, xmax,  652,  571, 'h'], #2
                            [xmin, xmax,  745,  816, 'h'], #3
                            [ 100,  100, ymax, ymin, 'v'], #4
                            [ 200,  200, ymax, ymin, 'v'], #5
                            [ 300,  300, ymax, ymin, 'v'], #6
                            [ 400,  400, ymax, ymin, 'v'], #7
                            [ 100,  500,  200,  200, 'p'] ]#8 Profile


        self.xmax = 1750
        self.ymax = 1750

        # For entire CSPad   Xmin  Xmax  Ymin  Ymax Orient
        self.line_coord = [ [xmin, 1678,  230,ymin, 'h'], #0# Horizontal wires
                            [xmin, xmax,  337, 371, 'h'], #1
                            [xmin, xmax,  652, 571, 'h'], #2
                            [xmin, xmax,  745, 816, 'h'], #3
                            [xmin, xmax,  969,1038, 'h'], #4
                            [xmin, xmax, 1229,1267, 'h'], #5
                            [xmin, xmax, 1416,1333, 'h'], #6
                            [xmin, xmax, 1625,1697, 'h'], #7
                            [xmin,  667,  726,ymin, 'v'], #8 Vertical wires
                            [xmin, 1220, 1472,ymin, 'v'], #9
                            [ 495, xmax, ymax, 110, 'v'], #10
                            [ 858, xmax, ymax, 262, 'v'], #11
                            [1215, xmax, ymax, 570, 'v'], #12
                            [1000, 1500,  500, 500, 'p'] ]#13 Profile


        self.logDirName     = '.'
        self.logFileName    = 'z-logfile.txt'

        self.imgDirName     = '.'
        self.imgFileName    = 'cspad-ave100-cxi80410-r0628-quad-2-v02.txt'
        self.imgDirName     = '../HDF5Explorer-v01'
        self.imgFileName    = 'cspad-ave.txt'

        self.configDirName  = '.'
        self.configFileName = 'z-configpars.txt'

  
        self.plot_fname_suffix = 'det'
        #self.plot_fname_suffix = 'quad-2'




        self.number_of_lines = len(self.line_coord)


    def setRunTimeParameters ( self ) :
        print 'setRunTimeParameters'
        self.selected_line_index = None


    def printImageArray ( self ) :
        print 'Current image array:\n', self.arr


    def printLineCoordinates ( self ) :
        print 'len(line_coord) =', len(self.line_coord)
        #print 'line_coord = ', self.line_coord 
        for lind in range(len(self.line_coord)) :
            print 'line %2.2d' % (lind),
            print '   coordinates =', self.line_coord[lind]  


    def printParameters ( self ) :
        """Prints current values of configuration parameters
        """
        print '\nImageParameters'

        print 'LOG_DIR_NAME',      self.logDirName   
        print 'LOG_FILE_NAME',     self.logFileName   
        print 'IMG_DIR_NAME',      self.imgDirName   
        print 'IMG_FILE_NAME',     self.imgFileName   
        print 'CONFIG_DIR_NAME',   self.configDirName   
        print 'CONFIG_FILE_NAME',  self.configFileName   
        print 'PLOT_FNAME_SUFFIX', self.plot_fname_suffix   

        print 'NUMBER_OF_LINES',   len(self.line_coord)  

        for lind in range(len(self.line_coord)) :

            print 'LINE_INDEX',    lind 
            print 'LINE_X1',       self.line_coord[lind][0] 
            print 'LINE_X2',       self.line_coord[lind][1] 
            print 'LINE_Y1',       self.line_coord[lind][2] 
            print 'LINE_Y2',       self.line_coord[lind][3] 
            print 'LINE_TYPE',     self.line_coord[lind][4]  



    def __setConfigParsFileName(self, fname=None) :
        if fname == None :
            self._fname = self.configDirName + '/' + self.configFileName
        else :
            self._fname = fname


    def readParameters(self, fname=None) :
        self.__setConfigParsFileName(fname)        
        print 'Read parameters from file:', self._fname
        dicBool = {'false':False, 'true':True}
        if os.path.exists(self._fname) :
            f=open(self._fname,'r')
            for line in f :
                if len(line) == 1 : continue # line is empty
                key = line.split()[0]
                val = line.split()[1]

                print 'key, val=', key, val

                if   key == 'LOG_DIR_NAME'     : self.logDirName          = val
                elif key == 'LOG_FILE_NAME'    : self.logFileName         = val
                elif key == 'IMG_DIR_NAME'     : self.imgDirName          = val
                elif key == 'IMG_FILE_NAME'    : self.imgFileName         = val
                elif key == 'CONFIG_DIR_NAME'  : self.configDirName       = val
                elif key == 'CONFIG_FILE_NAME' : self.configFileName      = val
                elif key == 'PLOT_FNAME_SUFFIX': self.plot_fname_suffix   = val
                elif key == 'NUMBER_OF_LINES'  : self.number_of_lines          = int(val)
                elif key == 'LINE_INDEX'       : self.lind                     = int(val)
                elif key == 'LINE_X1'          : self.line_coord[self.lind][0] = int(val)
                elif key == 'LINE_X2'          : self.line_coord[self.lind][1] = int(val)
                elif key == 'LINE_Y1'          : self.line_coord[self.lind][2] = int(val)
                elif key == 'LINE_Y2'          : self.line_coord[self.lind][3] = int(val)
                elif key == 'LINE_TYPE'        : self.line_coord[self.lind][4] = str(val)

            f.close()
        else :
            print 'The file %s does not exist' % (fname)
            print 'WILL USE DEFAULT CONFIGURATION PARAMETERS'

    def writeParameters(self, fname=None) :
        self.__setConfigParsFileName(fname)        
        print 'Write parameters in file:', self._fname
        space = '    '
        
        f=open(self._fname,'w')
        f.write('LOG_DIR_NAME'     + space + self.logDirName                   + '\n')
        f.write('LOG_FILE_NAME'    + space + self.logFileName                  + '\n')
        f.write('IMG_DIR_NAME'     + space + self.imgDirName                   + '\n')
        f.write('IMG_FILE_NAME'    + space + self.imgFileName                  + '\n')
        f.write('CONFIG_DIR_NAME'  + space + self.configDirName                + '\n')
        f.write('CONFIG_FILE_NAME' + space + self.configFileName               + '\n')
        f.write('PLOT_FNAME_SUFFIX'+ space + self.plot_fname_suffix            + '\n')
        f.write('\n')
        f.write('NUMBER_OF_LINES'  + space + str(len(self.line_coord))         + '\n')

        for lind in range(len(self.line_coord)) :
            f.write('\n')
            f.write('LINE_INDEX'   + space + str(lind)                         + '\n')
            f.write('LINE_X1'      + space + str(self.line_coord[lind][0] )    + '\n')
            f.write('LINE_X2'      + space + str(self.line_coord[lind][1] )    + '\n')
            f.write('LINE_Y1'      + space + str(self.line_coord[lind][2] )    + '\n')
            f.write('LINE_Y2'      + space + str(self.line_coord[lind][3] )    + '\n')
            f.write('LINE_TYPE'    + space +     self.line_coord[lind][4]      + '\n')

        f.close()


 
#---------------------------------------
# Makes a single object of this class --
#---------------------------------------

impars = ImageParameters()  
#impars.loadImageArrayFromFile ( 'cspad-ave100-cxi80410-r0628-v02.txt' )
#impars.printImageArray()
#impars.printLineCoordinates()

#----------------------------------
#def main():
#    sys.exit()    
#
#if __name__ == '__main__':
#    main()
#----------------------------------

