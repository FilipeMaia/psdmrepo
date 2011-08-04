import numpy as np
import matplotlib.pyplot as plt

class CsPad( object ):

    npix_quad = 850
    
    # origin of section in quad coordinate system
    #
    # x-position correspond to column number
    xpos_sec2x1 = [[ 414,  626,    0,    0,  213,    1,  418,  419],  # 2:5 were not measured
                   [ 421,  634,    0,    0,  213,    1,  424,  425],
                   [ 417,  630,    0,    1,  212,    0,  425,  426],
                   [ 416,  630,    0,    0,  213,    1,  420,  421]] # 2:5 were not measured
    # y-position correspond to maxrows - row number    
    ypos_sec2x1 = [[   0,    0,  214,    1,  425,  425,  615,  402],  # 2:5 were not measured
                   [   0,    0,  214,    1,  425,  425,  615,  402],
                   [   0,    0,  215,    3,  431,  431,  616,  403],
                   [   0,    0,  214,    1,  425,  425,  615,  403]] # 2:5 were not measured
    

    def __init__(self, config):
        quads = range(4)
        self.sections = map(config.sections, quads)
        pass

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

                
            # sections 2,3 and 6,7 are as is. The others need some rotation:
            if i==0 or i==1 :
                pair = pair[:,::-1].T
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

        pairs = []
        for i in range (8) :
        
            # insert gap between asics in the 2x1
            asics = np.hsplit( data3d[i], 2)
            gap = np.zeros( (185,4), dtype=data3d.dtype )
            #gap = gap * -99
            pair = np.hstack( (asics[0], gap, asics[1]) )

                
            # sections 2,3 and 6,7 are as is. The others need some rotation:
            if i==0 or i==1 :
                pair = pair[:,::-1].T
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



    def CsPadElementUnaligned( self, data3d, qn ):
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

        zeros = np.zeros((18,388),dtype=data3d.dtype)
        zeros9 = np.zeros((9,388),dtype=data3d.dtype)
        zeros6 = np.zeros((6,388),dtype=data3d.dtype)

        # if any sections are missing, insert zeros
        if len( data3d ) < 8 :
            zsec = np.zeros( (185,388), dtype=data3d.dtype)
            for i in range (8) :
                if i not in self.sections[qn] :
                    data3d = np.insert( data3d, i, zsec, axis=0 )
                #print "section ", i
                #print data3d[i]


        s01 = np.concatenate( (zeros6.T,
                               data3d[0][:,::-1].T,
                               zeros6.T,
                               data3d[1][:,::-1].T,
                               zeros6.T),
                              1)
        s23 = np.concatenate( (zeros6,
                               data3d[2], 
                               zeros6,
                               data3d[3],
                               zeros6 ),
                              0 )
        s45 = np.concatenate( (zeros6.T,
                               data3d[5][::-1,:].T,
                               zeros6.T,
                               data3d[4][::-1,:].T,
                               zeros6.T), 
                              1 )
        s67 = np.concatenate( (zeros6,
                               data3d[6], 
                               zeros6,
                               data3d[7],
                               zeros6 ),
                              0 )

        m1 = np.hstack( (s23, s01) )
        m2 = np.hstack( (s45, s67) )
        e0 = np.vstack( (m2, m1) )

        if qn>0 : e0 = np.rot90( e0, 4-qn)
        return e0

