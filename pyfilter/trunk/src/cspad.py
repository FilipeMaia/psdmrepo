import numpy as np
np.set_printoptions(precision=3,suppress=True)

import math
import scipy.ndimage.interpolation as interpol
import scipy.misc

import AppUtils.AppDataPath as apputils

class CsPadAssembler( object ):
    """CsPad class for creating CsPad image for an event within pyana
    """
    npix_quad = 2 * (388 + 3) + 68 # = 850 (68 is arbitrary padding)
    
    def __init__(self, devicename, config, calibfinder, run):
        """ Initialize CsPad object

        @param devicename  address of detector/device in xtc file
        @param config      configuration object
        @param calibfinder calibfinder object
        @param run         integer, run number
        """
        
        self.sections = map(config.sections,range(4))

        self.pedestals = None
        self.image = None 
        self.pixels = np.zeros((4,8,185,388), dtype="uint16")
        self.image = np.zeros((2*self.npix_quad+100, 2*self.npix_quad+100 ), dtype="float64")

        self.small_angle_tilt = False

        self.read_alignment(calibfinder,devicename,run)
        
        self.x_coordinates = None
        self.y_coordinates = None
        self.z_coordinates = None
        #self.make_coordinate_map()


    def read_alignment(self, calibfinder, devicename, runnr):
        """Alignment calibrations as defined for psana.
        Read in these standard parameter files. Alternative
        path/file can be given by arguments
        """
        # angle of each section = rotation (nx90 degrees) + tilt (small angles)
        # ... (4 rows (quads) x 8 columns (sections))
        self.rotation_array =  np.loadtxt( calibfinder.findCalibFile(devicename,"rotation",runnr))
        self.tilt_array     =  np.loadtxt( calibfinder.findCalibFile(devicename,"tilt",    runnr))
        self.section_angles = self.rotation_array + self.tilt_array
        
        # read in center position of sections in each quadrant (quadrant coordinates)
        # ... (3*4 rows (4 quads, 3 xyz coordinates) x 8 columns (sections))
        centers              = np.loadtxt( calibfinder.findCalibFile(devicename,"center",runnr))
        center_corrections   = np.loadtxt( calibfinder.findCalibFile(devicename,"center_corr",runnr))
        self.section_centers = np.reshape( centers + center_corrections, (3,4,8) )
        
        # read in the quadrant offset parameters (w.r.t. image 0,0 in upper left coner)
        # ... (3 rows (xyz) x 4 columns (quads) )
        quad_pos      = np.loadtxt(calibfinder.findCalibFile(devicename,"offset",runnr))
        quad_pos_corr = np.loadtxt( calibfinder.findCalibFile(devicename,"offset_corr",runnr))
        quad_position = quad_pos + quad_pos_corr


        # read in margins file:
        # ... (3 rows (x,y,z) x 4 columns (section offset, quad offset, quad gap, quad shift)
        marg_gap_shift = np.loadtxt( calibfinder.findCalibFile(devicename,"marg_gap_shift",runnr))

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


    def make_coordinate_map(self):
        """Make coordinate maps from meterology file
        """
        print "Making coordinate map of the CSPAD detector."
        self.x_coordinates = np.zeros((4,8,185,388), dtype="float")
        self.y_coordinates = np.zeros((4,8,185,388), dtype="float")
        self.z_coordinates = np.zeros((4,8,185,388), dtype="float")

        def get_asics(bigsection):
            """Utility function"""
            asic0 = bigsection[:,0:194]
            asic1 = bigsection[:,(391-194):]
            asics = np.concatenate( (asic0,asic1), axis=1 )
            return asics
        
        # section pixel array / grid
        rr,cc = np.mgrid[0:185:185j, 0:391:391j]
        
        # now compute the "fractional pixels"
        rrfrac = rr / 185.0
        ccfrac = cc / 391.0
        
        # remove the 3-pixel gap
        rrfrac = get_asics(rrfrac)
        ccfrac = get_asics(ccfrac)
        
        sec_coords = np.array([rrfrac,ccfrac])
        sec_coord_order = [(1,2,0,3),(1,2,0,3),(2,3,1,0),(2,3,1,0),(3,0,2,1),(3,0,2,1),(2,3,1,0),(2,3,1,0)]
        
        # load data from metrology file (ignore first column)
        metrology = np.loadtxt("data/XtcExplorer/calib/CSPad/cspad_2011-08-10-Metrology.txt")[:,1:]
        metrology = metrology.reshape(4,8,4,3)

        # also, we need to resort the 2x1 sections, they are
        # listed in the file in the order 1,0,3,2,4,5,7,6
        metrology = metrology[:,(1,0,3,2,4,5,7,6),:,:]

        dLong = np.zeros((4,8,2), dtype="float64")
        dShort = np.zeros((4,8,2), dtype="float64")
        for quad in range(4):

            for sec in range(8):
                
                # corner positions (in micrometers)
                input_x = metrology[quad,sec,sec_coord_order[sec],0].reshape(2,2)
                input_y = metrology[quad,sec,sec_coord_order[sec],1].reshape(2,2)
                input_z = metrology[quad,sec,sec_coord_order[sec],2].reshape(2,2)
                #print "X ", input_x
                #print "Y ", input_y
                #print "Z ", input_z
        
                # interpolate coordinates over to the pixel map
                self.x_coordinates[quad,sec] = interpol.map_coordinates(input_x, sec_coords)
                self.y_coordinates[quad,sec] = interpol.map_coordinates(input_y, sec_coords)
                self.z_coordinates[quad,sec] = interpol.map_coordinates(input_z, sec_coords)
                #print "x ", self.x_coordinates[quad,sec]
                #print "y ", self.y_coordinates[quad,sec]
                #print "z ", self.z_coordinates[quad,sec]
                
                # ! in micrometers! Need to convert to pixel units
                dL = np.array([ abs(input_x[0,1]-input_x[0,0])/391, 
                                abs(input_x[1,1]-input_x[1,0])/391,
                                abs(input_y[0,0]-input_y[0,1])/391,
                                abs(input_y[1,0]-input_y[1,1])/391 ])
                dLong[quad,sec] = dL[dL>100] # filter out the nonsense ones
                
                dS = np.array([ abs(input_y[0,0]-input_y[1,0])/185,
                                abs(input_y[0,1]-input_y[1,1])/185, 
                                abs(input_x[0,0]-input_x[1,0])/185,
                                abs(input_x[0,1]-input_x[1,1])/185 ])
                dShort[quad,sec] = dS[dS>100] # filter out the nonsense ones

        dTotal = np.concatenate( (dLong.ravel(), dShort.ravel() ))
        print "Pixel-size:"
        print "     long side average:    %.2f +- %.2f "%( dLong.mean(), dLong.std())
        print "     short side average:   %.2f +- %.2f "%( dShort.mean(), dShort.std())
        print "     all sides average:    %.2f +- %.2f "%( dTotal.mean(), dTotal.std())

        # use the total to convert it all to pixel units
        self.x_coordinates = self.x_coordinates / dTotal.mean()
        self.y_coordinates = self.y_coordinates / dTotal.mean()
        self.z_coordinates = self.z_coordinates / dTotal.mean()

        origin = [[834,834],[834,834],[834,834],[834,834]]
        for quad in range(4):
            # For each quad, rotate and shift into the image coordinate system
            if quad==0 :
                savex = np.array( self.x_coordinates[quad] )
                self.x_coordinates[quad] = origin[quad][0] - self.y_coordinates[quad]
                self.y_coordinates[quad] = origin[quad][1] - savex
            if quad==1 :
                self.x_coordinates[quad] = origin[quad][0] + self.x_coordinates[quad]
                self.y_coordinates[quad] = origin[quad][1] - self.y_coordinates[quad]
            if quad==2 :
                savex = np.array( self.x_coordinates[quad] )                
                self.x_coordinates[quad] = origin[quad][0] + self.y_coordinates[quad]
                self.y_coordinates[quad] = origin[quad][1] + savex
            if quad==3 :
                self.x_coordinates[quad] = origin[quad][0] - self.x_coordinates[quad]
                self.y_coordinates[quad] = origin[quad][1] + self.y_coordinates[quad]
                        

        print "Done making coordinate map of the CSPAD detector."
        #np.savetxt("xcoord.txt",self.x_coordinates.reshape((4*8*185,388)),fmt='%.1f')


    def load_pedestals(self, pedestalsfile ):
        """ load dark from pedestal file:
        pedestals txt file is (4*8*185)=5920 (lines) x 388 (columns)
        accepted file formats: ascii and npy (binary)
        """
        try: 
            if pedestalsfile.find(".npy") >= 0:
                self.pedestals = np.load(pedestalsfile).reshape((4,8,185,388))
            else :
                self.pedestals = np.loadtxt(pedestalsfile).reshape((4,8,185,388))
            print "Pedestals has been loaded from %s"% pedestalsfile
            print "Pedestals will be subtracted from displayed images"
        except:
            print "No pedestals loaded. File name requested was ", pedestalsfile
            pass


    def get_mini_image(self, element ):
        """get_2x2_image
        @param element      a single CsPad.MiniElementV1
        """
        data = element.data()   # 185x388x2
        
        self.pixels = np.array( (data[:,:,0],data[:,:,1]) )
        # pixels should now be (2 x 185 x 388)

        pairs = []
        for i in xrange(2):
            asics = np.split( data[:,:,i],2,axis=1)

            # gap should be 3 pixels wide
            gap = np.zeros( (185,3), dtype=data.dtype )
            pair = np.concatenate( (asics[0], gap, asics[1]), axis=1 )

            pair = pair[:,::-1].T
            pairs.append(pair)

        # wedge some pixels between the two 2x1s
        wedge = np.zeros( (391,10), data.dtype )
        image = np.hstack( (pairs[0],wedge,pairs[1]) )
        return image


    def get_detector_image(self, elements, filter=None ):
        """get_detector_image
        @param elements     list of CsPad.ElementV1 from pyana evt.getCsPadQuads()
        @param fiter        None, or tuple of (threshold,count)
        """
        self.pixels = self.get_pixel_array( elements )
        #print "pixel average ", self.pixels.mean()

        # pedestal subtraction here
        if self.pedestals is not None:
            self.pixels = self.pixels - self.pedestals
            #print "after ped subt ", self.pixels.mean()

            # common mode subtraction here (only if pedestal subtracted first)
            #for q in xrange(4):
            #    for s in xrange(8):
            #        array = self.pixels[q][s]
            #        self.pixels[q][s] = array - self.common_mode(array,30)
            #print "after cm subt ", self.pixels.mean()

            
        self.assemble_image( self.pixels )
        return self.image



    def get_pixel_array( self, elements ):
        """make pixel array (4,8,185,388) from pyana elements
        """             
        for e in elements: 
            data = e.data()
            quad = e.quad()
            self.pixels[quad] = self.complete_quad(data, quad)
        return self.pixels

    def subtract_pedestals( self ):
        try:
            self.pixels = self.pixels - self.pedestals
        except:
            if self.pixels is None: print "pixels not yet set up "
            if self.pedestals is None: print "pedestals not yet set up "
        return self.pixels

    def subtract_commonmode( self, threshold=30 ):
        if self.pedestals is None:
            print "No pedestals defined, cannot do common mode"
            return

        # this algorithm adds 58 ms / event
        array = self.pixels.reshape(32,185*388)
        cmmode = [ np.average(array[i][array[i]<threshold]) for i in xrange(0,32)]
        for i in xrange(32):
            array[i] = array[i]-cmmode[i]
        self.pixels = array.reshape(4,8,185,388)
        return self.pixels

        # this algorithm adds 121 ms / event
        array = self.pixels.reshape(32,185*388)
        cmmode = np.ma.average( np.ma.masked_array(array, mask=(array>=threshold) ), axis=1 )
        for i in xrange(32):
            array[i] = array[i]-cmmode[i]
        self.pixels = array.reshape(4,8,185,388)
        return self.pixels

        # this algorithm adds 162 ms / event
        for q in xrange(4):
            for s in xrange(8):
                section = self.pixels[q,s]
                cmmode =  np.average(section[section<threshold]) # avoid noisy pixels
                self.pixels[q,s] = self.pixels[q,s] - cmmode
        return self.pixels

            
    def assemble_image(self, data2d=None ):
        """Assemble an image from 2D data array file (e.g. pedestals file),
        or if none is given, assemble an image from self.pixels 
        """
        if data2d is not None:
            self.pixels = data2d.reshape(4,8,185,388)

        for quad in xrange (4):

            quad_image = self.assemble_quad_image( self.pixels[quad], quad )
            if quad>0:
                # reorient the quad_image as needed
                quad_image = np.rot90( quad_image, 4-quad)

            qoff_x = self.quad_offset[0,quad]
            qoff_y = self.quad_offset[1,quad]
            self.image[qoff_x:qoff_x+self.npix_quad, qoff_y:qoff_y+self.npix_quad]=quad_image

        # mask out hot/saturated pixels (16383)
        #im_hot_masked = np.ma.masked_greater_equal( self.image, 16383 )
        #self.image = np.ma.filled( im_hot_masked, 0)
        return self.image
         

    def complete_quad( self, data3d, qn ):
        # if any sections are missing, insert zeros
        if len( data3d ) < 8 :
            zsec = np.zeros( (185,388), dtype=data3d.dtype)
            for i in xrange (8) :
                if i not in self.sections[qn] :
                    data3d = np.insert( data3d, i, zsec, axis=0 )

        # now the sections have been "filled out", fill the pixels attribute
        return data3d



    def assemble_quad_image( self, data3d, qn) :
        """assemble_quad_image
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
        #
        # each section read from the event has "landscape" orientation
        # with 185 rows (first index) and 2*194 columns (2nd index)
        #   - Sections 0,1: "portrait" orientation / tilted 90 degrees counter clockwise:
        #                    first index increases from left to right, 2nd index increases from bottom to top, 
        #   - Sections 2,3: "landscape" orientation / as is:
        #                    first index increases from top to bottom, 2nd index increases from left to right
        #   - Sections 4,5: "portrait" orientation / tilted 90 degrees clockwise:
        #                    first index increases from right to left, 2nd index increases from top to bottom, 
        #   - Sections 6,7: "landscape" orientation / as is:
        #                    first index increases from top to bottom, 2nd index increases from left to right
        #   Again, the orientations of the Sections for quadrant 1 are rotated 90 degrees clockwise
    
        pairs = []
        for i in xrange (8) :
        
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


    def save_pixels(self, pixelsfile ):
        """ save image as a pixel array with the same 
        format as the pedestals txt file: 5920 (lines) x 388 (columns)
        accepted file formats: ascii and npy (binary)
        """
        try: 
            if pixelsfile.find(".npy") >= 0:
                np.save(pixelsfile, self.pixels.reshape((5920,388)))
            else :
                np.savetxt(pixelsfile, self.pixels.reshape((5920,388)))
            print "Pixels have been saved to %s"% pixelsfile
        except:
            print "Could not save to file ", pixelsfile
            pass


    def save_image(self, imagefile ):
        """ save image
        """
        try: 
            if imagefile.find(".npy") >= 0:
                np.save(imagefile, self.image)
            elif  imagefile.find(".txt") >= 0:
                np.savetxt(imagefile, self.image)
            else :
                scipy.misc.imsave(imagefile, self.image)
            print "Image has been saved to %s"% imagefile
        except:
            print "Could not save to file ", imagefile
            pass

