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

    def __init__(self, sections, path = None):
        """ Initialize CsPad object
        @param sections    list of active sections from the cfg object
        @param path        optional, path to calibration directory
        """
        self.sections = sections

        self.pedestals = None
        # one array 4 x 8 x 185 x 388
        # (flattens to 5920 x 388)

        self.quad_arrays = [None, None, None, None]
        # 4 arrays, each (8 x 185 x 388)

        self.pixel_array = None
        # one array 4 x 8 x 185 x 388
        # (flattens to 5920 x 388)

        self.image = None 
        # one 1800x1800 image ready for display

        self.read_geometry(path)

    def set_pedestals(self, pedestalfile ):

        # load dark from pedestal file:
        # pedestals txt file is (4*8*185)=5920 (lines) x 388 (columns)
        try: 
            self.pedestals = np.loadtxt(pedestalfile)
            self.pedestals = np.reshape(4,8,185,388)
            print "Pedestals has been loaded from %s"% pedestalfile
            print "Pedestals will be subtracted from displayed images"
        except:
            print "No pedestals loaded"
            pass
         

    def read_geometry(self, path = None):
        """
        Geometry calibrations are defined the same as for psana.
        Read in these standard parameter files.
        """
        if path is None: 
            path = 'XtcExplorer/calib/CsPad/'

        # read in rotation array:
        # 90-degree angle orientation of each section in each quadrant. 
        # ... (4 rows (quads) x 8 columns (sections))
        self.rotation_array =  np.loadtxt('XtcExplorer/calib/CSPad/rotation.par')

        # read in tilt array
        # ... (4 rows (quads) x 8 columns (sections))
        self.tilt_array =  np.loadtxt('XtcExplorer/calib/CSPad/tilt.par')

        # read in center position of sections in each quadrant
        # ... (3*4 rows (4 quads, 3 xyz coordinates) x 8 columns (sections))
        ctr_array = np.loadtxt('XtcExplorer/calib/CSPad/center.par')
        ctr_corr_array = np.loadtxt('XtcExplorer/calib/CSPad/center_corr.par')
        self.center_array = np.reshape( ctr_array + ctr_corr_array, (3,4,8) )

        # read in the quadrant offset parameters (w.r.t. image 0,0 in upper left coner)
        # ... (3 rows (xyz) x 4 columns (quads) )
        quad_pos = np.loadtxt('XtcExplorer/calib/CSPad/offset.par')
        quad_pos_corr = np.loadtxt('XtcExplorer/calib/CSPad/offset_corr.par')
        quad_position = quad_pos[0:2,:] + quad_pos_corr[0:2,:]

        
        # read in margins file:
        # ... (3 rows (x,y,z) x 4 columns (section offset, quad offset, quad gap, quad shift)
        marg_gap_shift = np.loadtxt('XtcExplorer/calib/CSPad/marg_gap_shift.par')

        # break it down (extract each column, make full arrays to be added to the above ones)
        self.sec_offset = marg_gap_shift[0:2,0]

        quad_offset_xy = marg_gap_shift[0:2,1]
        quad_gap_xy = marg_gap_shift[0:2,2]
        quad_shift_xy = marg_gap_shift[0:2,3]
        
        quad_offset_XY = np.array( [quad_offset_xy,
                                    quad_offset_xy,
                                    quad_offset_xy,
                                    quad_offset_xy] ).T
        quad_gap_XY = np.array( [quad_gap_xy*[-1,-1],
                                 quad_gap_xy*[-1,1],
                                 quad_gap_xy*[1,1],
                                 quad_gap_xy*[1,-1]] ).T
        quad_shift_XY = np.array( [quad_shift_xy*[1,-1],
                                   quad_shift_xy*[-1,-1],
                                   quad_shift_xy*[-1,1],
                                   quad_shift_xy*[1,1]] ).T
        self.quad_offset = quad_position + quad_offset_XY + quad_gap_XY + quad_shift_XY

        

    def complete( self, data3d, qn ):
        # if any sections are missing, insert zeros
        if len( data3d ) < 8 :
            zsec = np.zeros( (185,388), dtype=data3d.dtype)
            #zsec = zsec * -99
            for i in range (8) :
                if i not in self.sections[qn] :
                    data3d = np.insert( data3d, i, zsec, axis=0 )

        # now the sections have been "filled out", fill the pixels attribute
        self.quad_arrays[qn] = data3d.reshape(1480,388)
        return data3d

    def make_pixel_array( self, elements ):

        self.pixel_array = np.zeros((4,8,185,388), dtype="uint16")
        for e in elements: 
            data = e.data()
            quad = e.quad()
            
            data = self.complete(data, quad)
            self.pixel_array[quad] = data


    def get_quad_image( self, data3d, qn ):

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
#            pair = interpol.rotate(pair,self.tilt_array[qn][i])
#            print "shape of pair after rotation", pair.shape

        # make the array for this quadrant
        quadrant = np.zeros( (self.npix_quad, self.npix_quad), dtype=data3d.dtype )

        # insert the 2x1 sections according to
        for sec in range (8):
            nrows, ncols = pairs[sec].shape

            # colp,rowp are where the corner of a section should be placed
            rowp = self.npix_quad - (self.sec_offset[0] + self.center_array[0][qn][sec] - nrows/2)
            colp = self.npix_quad - (self.sec_offset[1] + self.center_array[1][qn][sec] + ncols/2)

            #if sec < 2 :
            quadrant[rowp-nrows:rowp, colp:colp+ncols] = pairs[sec][0:nrows,0:ncols]
            if (rowp > self.npix_quad) or (colp+ncols > self.npix_quad) :
                print "ERROR"
                print rowp-nrows, ":", rowp, ", ", colp, ":",colp+ncols

        # Finally, reorient the quadrant as needed
        if qn>0 : quadrant = np.rot90( quadrant, 4-qn)

        # Tilt quadrant too?

        return quadrant


    def get_mini_image(self, element ):
        """get_2x2_image
        @param element      a single CsPad.MiniElementV1
        """
        data = element.data()
        quad = element.quad()
        self.pixel_array = np.array( (data[:,:,0],data[:,:,1]) )
        # pixels should now be (2 x 185 x 388)

        pairs = []
        for i in xrange(2):
            asics = np.split( data[:,:,i],2,axis=1)
            gap = np.zeros( (185,3), dtype=data.dtype )
            # gap should be 3 pixels wide
            pair = np.concatenate( (asics[0], gap, asics[1]), axis=1 )
            pairs.append(pair)
        image = np.vstack( (pairs[0],pairs[1]) )
        return image

    def get_detector_image(self, elements ):
        """get_detector_image
        @param elements     list of CsPad.ElementV1 from pyana evt.getCsPadQuads()
        """
        self.make_pixel_array( elements )

        # pedestal subtraction here
        if self.pedestals is not None:
            self.pixel_array = self.pixel_array - self.pedestals

        cspad_image = np.zeros((2*self.npix_quad+100, 2*self.npix_quad+100 ), dtype="uint16")
        for quad in xrange (4):

            quad_image = self.get_quad_image( self.pixel_array[quad], quad )

            qoff_x = self.quad_offset[0,quad]
            qoff_y = self.quad_offset[1,quad]
            cspad_image[qoff_x:qoff_x+self.npix_quad, qoff_y:qoff_y+self.npix_quad]=quad_image


        # mask out hot/saturated pixels (16383)
        im_hot_masked = np.ma.masked_greater_equal( cspad_image,16383 )
        cspad_image = np.ma.filled( im_hot_masked, 0)

        return cspad_image



    def quad_imageCommonModeCorr( self, data3d, qn ):
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


