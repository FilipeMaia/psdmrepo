import numpy as np
import math
import scipy.ndimage.interpolation as interpol

class CsPad( object ):
    """CsPad class for creating CsPad image for an event within pyana
    """
    npix_quad = 2 * (388 + 3) + 68 # = 850 (68 is arbitrary padding)
    
    def __init__(self, sections = [[],[],[],[]], path = None):
        """ Initialize CsPad object
        @param sections    list of active sections from the cfg object
        @param path        optional, path to calibration directory
        """
        self.sections = sections

        self.pedestals = None
        self.pixels = None
        # one array 4 x 8 x 185 x 388
        # (flattens to 5920 x 388)

        self.qimages = [None, None, None, None]
        # 4 arrays, each (8 x 185 x 388)

        self.image = None 
        # one 1800x1800 image ready for display

        self.small_angle_tilt = False
        # apply additional small-angle tilt (in addition to
        # the 90-degree rotation of the sections). This improves
        # the image visually, but requires interpolation, 
        # and makes the display rather slow... 
        
        
        self.read_alignment(path)



    def read_alignment(self, path = None, file=None):
        """Alignment calibrations as defined for psana.
        Read in these standard parameter files. Alternative
        path/file can be given by arguments
        """
        if path is None: # use a local copy
            path = 'XtcExplorer/calib/CSPad'

        if file is None: # assume same file for all runs
            file = '0-end.data'
            
        # angle of each section = rotation (nx90 degrees) + tilt (small angles)
        # ... (4 rows (quads) x 8 columns (sections))
        self.rotation_array =  np.loadtxt('%s/rotation/%s'%(path,file))
        self.tilt_array =  np.loadtxt('%s/tilt/%s'%(path,file))
        self.section_angles = self.rotation_array + self.tilt_array
        
        # read in center position of sections in each quadrant (quadrant coordinates)
        # ... (3*4 rows (4 quads, 3 xyz coordinates) x 8 columns (sections))
        centers = np.loadtxt('%s/center/%s'%(path,file))
        center_corrections = np.loadtxt('%s/center_corr/%s'%(path,file))
        self.section_centers = np.reshape( centers + center_corrections, (3,4,8) )
        
        # read in the quadrant offset parameters (w.r.t. image 0,0 in upper left coner)
        # ... (3 rows (xyz) x 4 columns (quads) )
        quad_pos = np.loadtxt('%s/offset/%s'%(path,file))
        quad_pos_corr = np.loadtxt('%s/offset_corr/%s'%(path,file))
        quad_position = quad_pos + quad_pos_corr


        # read in margins file:
        # ... (3 rows (x,y,z) x 4 columns (section offset, quad offset, quad gap, quad shift)
        marg_gap_shift = np.loadtxt('%s/marg_gap_shift/%s'%(path,file))

        # break it down (extract each column, make full arrays to be added to the above ones)
        self.sec_offset = marg_gap_shift[:,0]

        quad_offset = marg_gap_shift[:,1]
        quad_gap = marg_gap_shift[:,2]
        quad_shift = marg_gap_shift[:,3]

        # turn them into 2D arrays
        quad_offset = np.array( [quad_offset,
                                 quad_offset,
                                 quad_offset,
                                 quad_offset] ).T
        quad_gap = np.array( [quad_gap*[-1,-1,1],     # numpy element-wise multiplication
                              quad_gap*[-1,1,1],
                              quad_gap*[1,1,1],
                              quad_gap*[1,-1,1]] ).T
        quad_shift = np.array( [quad_shift*[1,-1,1],
                                quad_shift*[-1,-1,1],
                                quad_shift*[-1,1,1],
                                quad_shift*[1,1,1]] ).T
        self.quad_offset = quad_position + quad_offset + quad_gap + quad_shift



    def get_mini_image(self, element ):
        """get_2x2_image
        @param element      a single CsPad.MiniElementV1
        """
        data = element.data()
        quad = element.quad()
        self.pixels = np.array( (data[:,:,0],data[:,:,1]) )
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
        self.pixels = self.make_pixel_array( elements )

        # pedestal subtraction here
        if self.pedestals is not None:
            self.pixels = self.pixels - self.pedestals

        self.make_image( self.pixels )
        return self.image



    def make_pixel_array( self, elements ):
        """make pixel array (4,8,185,388) from pyana elements
        """ 
        pixel_array = np.zeros((4,8,185,388), dtype="uint16")
        for e in elements: 
            data = e.data()
            quad = e.quad()
            pixel_array[quad] = self.complete_quad(data, quad)
        return pixel_array



    def make_image(self, data2d ):
        """make image
        
        Takes 'flattened' data array, applies the Cspad geometry 
        and returns the 1800x1800 image with all sections in the right place (or approximately)
        @param data2d  input 2d data array file (4*8*185 x 388 = 5920 x 388)
        """
        self.pixels = data2d.reshape(4,8,185,388)

        self.image = np.zeros((2*self.npix_quad+100, 2*self.npix_quad+100 ), dtype="uint16")
        for quad in xrange (4):

            quad_image = self.get_quad_image( self.pixels[quad], quad )
            self.qimages[quad] = quad_image
            if quad>0:
                # reorient the quad_image as needed
                quad_image = np.rot90( quad_image, 4-quad)

            qoff_x = self.quad_offset[0,quad]
            qoff_y = self.quad_offset[1,quad]
            self.image[qoff_x:qoff_x+self.npix_quad, qoff_y:qoff_y+self.npix_quad]=quad_image

        # mask out hot/saturated pixels (16383)
        im_hot_masked = np.ma.masked_greater_equal( self.image, 16383 )
        self.image = np.ma.filled( im_hot_masked, 0)
        return self.image
         
    def load_pedestals(self, pedestalfile ):
        """ load dark from pedestal file:
        pedestals txt file is (4*8*185)=5920 (lines) x 388 (columns)
        accepted file formats: ascii and npy (binary)
        """
        try: 
            if pedestalsfile.find(".npy") >= 0:
                self.pedestals = np.load(pedestalfile).reshape((4,8,185,388))
            else :
                self.pedestals = np.loadtxt(pedestalfile).reshape((4,8,185,388))
            print "Pedestals has been loaded from %s"% pedestalfile
            print "Pedestals will be subtracted from displayed images"
        except:
            print "No pedestals loaded. File name requested was ", pedestalsfile
            pass


    def complete_quad( self, data3d, qn ):
        # if any sections are missing, insert zeros
        if len( data3d ) < 8 :
            zsec = np.zeros( (185,388), dtype=data3d.dtype)
            #zsec = zsec * -99
            for i in range (8) :
                if i not in self.sections[qn] :
                    data3d = np.insert( data3d, i, zsec, axis=0 )

        # now the sections have been "filled out", fill the pixels attribute
        self.qimages[qn] = data3d.reshape(1480,388)
        return data3d



    def get_quad_image( self, data3d, qn) :
        """get_quad_image
        Get an image for this quad (qn)

        @param data3d           3d data array (row vs. col vs. section)
        @param qn               quad number
        """
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
            if self.small_angle_tilt :
                pair = interpol.rotate(pair,self.tilt_array[qn][i])
                #print "shape of pair after rotation", pair.shape

        # make the array for this quadrant
        quadrant = np.zeros( (self.npix_quad, self.npix_quad), dtype=data3d.dtype )

        # insert the 2x1 sections according to
        for sec in range (8):
            nrows, ncols = pairs[sec].shape

            # colp,rowp are where the top-left corner of a section should be placed
            rowp = self.npix_quad - self.sec_offset[0] - (self.section_centers[0][qn][sec] + nrows/2)
            colp = self.npix_quad - self.sec_offset[1] - (self.section_centers[1][qn][sec] + ncols/2)
            
            quadrant[rowp:rowp+nrows, colp:colp+ncols] = pairs[sec][0:nrows,0:ncols]
            if (rowp+nrows > self.npix_quad) or (colp+ncols > self.npix_quad) :
                print "ERROR"
                print rowp, ":", rowp+nrows, ", ", colp, ":",colp+ncols


        return quadrant



