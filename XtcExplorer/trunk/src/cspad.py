import numpy as np
import matplotlib.pyplot as plt

import scipy.ndimage.interpolation as interpol

class CsPad( object ):

    npix_quad = 2 * (388 + 3) + 68 # = 850 (68 is arbitrary padding)
    
    # Old numbers translated into New coordinate system: 
    # x-position correspond to column number
    xpos_sec2x1 = np.array(
        [[  0,   0, 214,   1, 425, 425, 615, 402],
         [  0,   0, 214,   1, 425, 425, 615, 402],
         [  0,   0, 215,   3, 431, 431, 616, 403],
         [  0,   0, 214,   1, 425, 425, 615, 403]]
        )
    ypos_sec2x1 = np.array(
        [[436, 224, 850, 850, 637, 849, 432, 431],
         [429, 216, 850, 850, 637, 849, 426, 425],
         [433, 220, 850, 849, 638, 850, 425, 424],
         [434, 220, 850, 850, 637, 849, 430, 429]]
        )

    def __init__(self, sections):
        self.sections = sections
        self.pedestals = None

        self.pixels = np.zeros((4*8*185,388), dtype="uint16")
        self.read_geometry()

    def set_pedestals(self, pedestalfile ):

        # load dark from pedestal file:
        # pedestals txt file is (4*8*185)=5920 (lines) x 388 (columns)
        try: 
            array = np.loadtxt(pedestalfile)
            self.pedestals = np.reshape(array, (4,8,185,388) )
            print "Pedestals has been loaded from %s"% pedestalfile
            print "Pedestals will be subtracted from displayed images"
        except:
            print "No pedestals loaded"
            pass
         

    def read_geometry(self):

        print self.xpos_sec2x1
        print self.ypos_sec2x1

#        # read in rotation array:
#        # 90-degree angle orientation of each section in each quadrant. 
#        self.rotation_array =  np.loadtxt('XtcExplorer/calib/CSPad/rotation.par')
#        print "Rotation array: ", self.rotation_array

        # read in tilt array
        self.tilt_array =  np.loadtxt('XtcExplorer/calib/CSPad/tilt.par')
        print "Tilt array: ", self.tilt_array

        # read in center position of sections in each quadrant
        ctr_array = np.loadtxt('XtcExplorer/calib/CSPad/center.par')
        ctr_corr_array = np.loadtxt('XtcExplorer/calib/CSPad/center_corr.par')
        # KIS (Keep It Simple)
        self.center_array = np.reshape( ctr_array + ctr_corr_array, (3,4,8) )

        print "Center array: ", self.center_array[0:2]
        #print "X = ", self.center_array[0]
        #print "Y = ", self.center_array[1]
        #print "Z = ", self.center_array[2]

        det_offset = np.loadtxt('XtcExplorer/calib/CSPad/offset.par')
        det_offset_corr = np.loadtxt('XtcExplorer/calib/CSPad/offset_corr.par')
        self.offset = det_offset + det_offset_corr
        print self.offset.shape

    def CsPadElement( self, data3d, qn ):

        # Construct one image for each quadrant, each with 8 sections
        # from a data3d = 3 x 2*194 x 185 data array
        #   +---+---+-------+
        #   |   |   |   6   |
        #   + 5 | 4 +-------+
        #   |   |   |   7   |
        #   +---+---+---+---+
        #   |   2   |   |   |
        #   +-------+ 0 | 1 |
        #   |   3   |   |   |
        #   +-------+---+---+

        # min and max
        #print "CsPad (min,max) for quad %d: (%d,%d)" % (qn,np.min(data3d),np.max(data3d))

        # if any sections are missing, insert zeros
        if len( data3d ) < 8 :
            zsec = np.zeros( (185,388), dtype=data3d.dtype)
            #zsec = zsec * -99
            for i in range (8) :
                if i not in self.sections[qn] :
                    data3d = np.insert( data3d, i, zsec, axis=0 )

                
        # subtract pedestals, if supplied --- !!UNTESTED!!
        if self.pedestals :
            print "Subtracting pedestals:"
            print "data3d has shape ", data3d.shape
            print "pedestals has shape ", self.pedestals
            print data3d
            print self.pedestals
            data3d -= self.pedestals


        pairs = []
        for i in range (8) :
        
            # 1) insert gap between asics in the 2x1
            asics = np.hsplit( data3d[i], 2)
            gap = np.zeros( (185,3), dtype=data3d.dtype )
            #
            # gap should be 3 pixels wide
            pair = np.hstack( (asics[0], gap, asics[1]) )

            # all sections are originally 185 (rows) x 388 (columns) 
            # Re-orient each section in the quad
            if i==0 or i==1 :
                pair = pair[:,::-1].T   # reverse columns, switch columns to rows. 
            if i==4 or i==5 :
                pair = pair[::-1,:].T   # reverse rows, switch rows to columns
            pairs.append( pair )

            # if tilt... 
            pair = interpol.rotate(pair,self.tilt_array[qn][i])
            print "shape of pair after rotation", pair.shape

        # make the array for this quadrant
        quadrant = np.zeros( (self.npix_quad, self.npix_quad), dtype=data3d.dtype )

        # insert the 2x1 sections according to
        for sec in range (8):
            nrows, ncols = pairs[sec].shape

            # colp,rowp are where the corner of a section should be placed
            colp = self.npix_quad - (self.center_array[1][qn][sec] + ncols/2)
            rowp = self.npix_quad - (self.center_array[0][qn][sec] - nrows/2)
            print "Quad#%d Section %d in  col=%d, row=%d" %(qn,sec,colp,rowp)

            if sec < 2 :
                quadrant[rowp-nrows:rowp, colp:colp+ncols] = pairs[sec][0:nrows,0:ncols]


        # Finally, reorient the quadrant as needed
        if qn>0 : quadrant = np.rot90( quadrant, 4-qn)

        # Tilt quadrant too?

        return quadrant


    def CsPad2x2Image(self, element ):
        """CsPad2x2Image
        @param element      a single CsPad.MiniElementV1
        """
        data = element.data()
        quad = element.quad()

        pairs = []
        for i in xrange(2):
            asics = np.split( data[:,:,i],2,axis=1)
            gap = np.zeros( (185,3), dtype=data.dtype )
            # gap should be 3 pixels wide
            pair = np.concatenate( (asics[0], gap, asics[1]), axis=1 )
            pairs.append(pair)
        image = np.vstack( (pairs[0],pairs[1]) )
        return image

    def CsPadImage(self, elements ):
        """CsPadImage
        @param elements     list of CsPad.ElementV1 from pyana evt.getCsPadQuads()
        """
        quad_images = np.zeros((4, self.npix_quad, self.npix_quad ), dtype="uint16")

        pixel_array = np.zeros( (4,8,185,388), dtype="uint16")

        for e in elements: 
            data = e.data()
            quad = e.quad()

            pixel_array[quad] = e.data()

            quad_images[quad] = self.CsPadElement( data, quad)
            

        self.pixels = pixel_array.reshape(5920,388)

        cspad_image = np.zeros((2*self.npix_quad, 2*self.npix_quad ), dtype="uint16")
        
        for qn in xrange (0,4):
            offset_x = self.offset[0][qn]
            offset_y = self.offset[1][qn]
            print offset_x
            print offset_y
            #cspad_image[offset_x:offset_x+self.npix_quad,
            #            offset_y, offset_y+self.npix_quad ] = quad_images[qn]
        
        return cspad_image




    def CsPadElementCommonModeCorr( self, data3d, qn ):
        # Construct one image for each quadrant, each with 8 sections
        # from a data3d = 3 x 2*194 x 185 data array
        #   +---+---+-------+
        #   |   |   |   6   |
        #   + 5 | 4 +-------+
        #   |   |   |   7   |
        #   +---+---+---+---+
        #   |   2   |   |   | 
        #   +-------+ 0 | 1 |
        #   |   3   |   |   |
        #   +-------+---+---+  

        # min and max
        #print "CsPad (min,max) for quad %d: (%d,%d)" % (qn,np.min(data3d),np.max(data3d))


        # if any sections are missing, insert zeros
        if len( data3d ) < 8 :
            zsec = np.zeros( (185,388), dtype=data3d.dtype)
            #zsec = zsec * -99
            for i in range (8) :
                if i not in self.sections[qn] :
                    data3d = np.insert( data3d, i, zsec, axis=0 )

        pairs = []
        for i in range (8) :

            # histogram of all pixel values (intensities) in this section
            nbins = 1000
            mean = np.mean( data3d[i] )
            std = np.std( data3d[i] )
            #low_high = ( np.min(data3d[i]), np.max(data3d[i]) )
            low_high = mean - 5*std, mean + 5*std
            #(hist, bin_edges) = np.histogram( data3d[i], nbins, low_high )
            #print hist
            #plt.plot(bin_edges[:-1],hist)
            plt.hist(np.ravel(data3d[i]),bins=nbins,range=low_high)
            plt.show()
        
            # insert gap between asics in the 2x1
            asics = np.hsplit( data3d[i], 2)
            gap = np.zeros( (185,4), dtype=data3d.dtype )
            #gap = gap * -99
            pair = np.hstack( (asics[0], gap, asics[1]) )

            # orientation of each section
            # ready to be inserted into the QuadElement array 
            # sections 2,3 and 6,7 are as is. The others need some rotation:
            if i==0 or i==1 :
                pair = pair[:,::-1].T   # rows as is, columns reversed, transpose
            if i==4 or i==5 :
                pair = pair[::-1,:].T

            pairs.append( pair )


        # make the array for this quadrant
        quadrant = np.zeros( (self.npix_quad, self.npix_quad), dtype=data3d.dtype )
        #quadrant = quadrant * -99

        # insert the 2x1 sections according to
        for sec in range (8):
            nrows, ncols = pairs[sec].shape

            # x,y  in quadrant coordinate system
            xpos = self.xpos_sec2x1[qn][sec]
            ypos = self.ypos_sec2x1[qn][sec]
            colp = xpos
            rowp = self.npix_quad-ypos

            quadrant[rowp-nrows:rowp, colp:colp+ncols] = pairs[sec][0:nrows,0:ncols]


        # Finally, rotate the quadrant as needed
        if qn>0 : quadrant = np.rot90( quadrant, 4-qn)
        return quadrant


