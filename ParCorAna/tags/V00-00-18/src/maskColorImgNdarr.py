import os
import sys
import numpy as np
import math
import psana
import scipy.ndimage
from PSCalib.CalibFileFinder import CalibFileFinder
from PSCalib.GeometryAccess import GeometryAccess, img_from_pixel_arrays
import scipy.ndimage as ndimage
import PsanaUtil

def debugOut(flag, msg):
    if flag:
        sys.stderr.write(msg)

def arrIndexListsToPoints(indexLists):
    '''helper function that takes a list of indicies and returns points

    Examples:
      >>> arrIndexListsToPoints([1,2,3],[0,1,1])
      [(1,0), (2,1), (3,1)]
    '''
    points = []
    numDims = len(indexLists)
    numPoints = len(indexLists[0])
    for idxList in indexLists[1:]:
        assert len(idxList)==numPoints
    for idx in range(numPoints):
        thisPoint = [indexLists[dim][idx] for dim in range(numDims)]
        points.append(thisPoint)
    return points

def makeColorArray(array, numColor):
    '''An example of making a color array based on intensity of values. 
    That is, if numColor=5, the bottom 20% of array will get 1, and the top
    20% will get 5.

    Returns array of same size with colors. 
    '''
    assert numColor>0
    sortedInds = np.argsort(array.flatten())
    N = len(sortedInds)
    if numColor > N:
        numColor = N
        print "warning: reduced numColor to %d for small amount of data" % numColor

    coloredFlat = np.zeros(N, np.int32)
    ind0 = 0
    delta = max(1,int(math.floor(N/numColor)))
    for color in range(1,numColor+1):
        ind1=min(N,ind0+delta)
        coloredFlat[sortedInds[ind0:ind1]]=color
        ind0 = ind1
    if ind1 < N:
        coloredFlat[sortedInds[ind1:]]=numColor
    coloredArray = coloredFlat.reshape(array.shape)
    return coloredArray

def findImageGaps(iX, iY):
    '''attempts to find gaps in maped image.
    
    are there any missed pixels in the image? Of course there is the
    space between the detector segments, the background, but within the
    segments, are there any holes? Label connected components of the pixels
    not mapped to by iX, iY. Report any components that do not touch the 
    border pixels.

    Returns: 
      a list of index lists, each the result of np.where(X) where
      X is in image space. These lists are for pixels where a believed 
      component is - i.e, a component is pixels
      in the image that are both not mapped to by iX,iY and do not touch
      a border pixel in the image
    '''

    imageShape = (np.int(np.max(iX) + 1),  # number of rows 
                  np.int(np.max(iY) + 1))  # number of columns

    image = np.ones(imageShape, np.int8)
    border = image.copy()
    border[:]=0
    border[0,:]=1
    border[-1,:]=1
    border[:,0]=1
    border[:,-1]=1
    border = border==1

    numRows, numCols = imageShape
    flatImageShape = (numRows * numCols,)
    image.shape = flatImageShape
    flatImageIndices = iX * numCols + iY
    image[flatImageIndices.flatten()]=0
    image.shape = imageShape

    gapComponents = []
    labeledArray, numLabels = ndimage.label(image)
    for labelIndex in range(numLabels):
        label = labelIndex + 1
        labeledBool = labeledArray == label
        numInComponent = np.sum(labeledBool)
        assert numInComponent > 0, "label=%d didn't label any pixels in image" % label
        hitsBorder = np.sum(labeledBool & border) > 0
        if not hitsBorder:
            gapComponents.append(np.where(labeledBool==True))
    return gapComponents
    
def findImageManyToOne(iX, iY):
    '''are there any pixels that have several ndarray elements mapped to them?    
    ::

      Returns:
      dict: keys are 2D image index pairs, 
            values are lists of ndarrays for the multi dimensional indicies of the ndarr
      i.e, return result might be:
      { (3,4):[ array(3,6,6), 
                array(4,5,1),
                array(0,1,2)] }
       meaning that ndarray elements  (3,4,0)  and (6,5,1)  and (6,1,2) all get mapped to
       the image position (3,4)
    '''
    flat_iX = iX.flatten()
    flat_iY = iY.flatten()
    pos2ndarr = {}
    ndarrFlatIdx = -1
    for a,b in zip(flat_iX, flat_iY):
        ndarrFlatIdx += 1
        imagePos = (a,b)
        if imagePos in pos2ndarr:
            pos2ndarr[imagePos].append(ndarrFlatIdx)
        else:
            pos2ndarr[imagePos] = [ndarrFlatIdx]

    manyToOneImagePos = dict([(imagePos, ndarrMappedTo) \
                              for imagePos,ndarrMappedTo in pos2ndarr.iteritems() \
                              if len(ndarrMappedTo)>1 ])
    newManyToOneImagePos = {}
    for imagePos, flatNdarrIndicies in manyToOneImagePos.iteritems():
        newManyToOneImagePos[imagePos] = np.unravel_index(flatNdarrIndicies,
                                                          iX.shape)
    return newManyToOneImagePos


class NoGeomFile(Exception):
    pass
        
def getGeometryFile(dsetstring, srcString, typeString):
    dataRoot = '/reg/d/psdm'
    assert os.path.exists(dataRoot), "data base directory=%s does not exist" % dataRoot
    dsetParts = PsanaUtil.parseDataSetString(dsetstring)
    calibDir = os.path.join(dataRoot, dsetParts['instr'], dsetParts['exp'], 'calib')
    assert os.path.exists(calibDir), "The experiment calib directory: %s doesn't exist" % calibDir
    typeNamespace = typeString.split('psana.')[1].split('.')[0]
    group = '%s::CalibV1' % typeNamespace
    def cleanSourceString(srcString):
        if srcString.find('(')==-1: return srcString
        return srcString.split('(')[1].split(')')[0]
    srcDir   = cleanSourceString(srcString)
    cf = CalibFileFinder(calibDir, group)
    run = dsetParts['run'][0]
    geomFile = cf.findCalibFile(srcDir, 'geometry', run)
    if not os.path.exists(geomFile):
        raise NoGeomFile(("geometry file %s doesn't exist.\n" + \
                         "Parameters: group=%s src=%s run=%s calibDir=%s\n" + \
                         "Note: geometry file may not be deployed, or inferred group is wrong Check calibDir.") % \
                         (geomFile, group, srcDir, run, calibDir))
    return geomFile

def ndarr2img(ndarr, iX, iY, backgroundValue = None):
    numRows = np.int(np.max(iX)+1)
    numCols = np.int(np.max(iY)+1)
    imageShape = (numRows, numCols)
    image = np.zeros(imageShape, ndarr.dtype)
    if backgroundValue is not None:
        image[:] = backgroundValue
    image[iX.flatten()[:], iY.flatten()[:]] = ndarr.flatten()[:]
    return image

def img2ndarr(img, iX, iY):
    '''converts an image to an ndarr
    '''
    numRows, numCols = img.shape
    ndarrShape = iX.shape
    ndarr = np.zeros(ndarrShape, img.dtype)
    iX_flat = iX.flatten()
    iY_flat = iY.flatten()
    if np.min(iX_flat) < 0:
        sys.stderr.write("WARNING: img2ndarr: iX has %d values < 0, clamping to 0\n" % np.sum(iX_flat<0))
        iX_flat[iX_flat < 0]=0
    if np.min(iY_flat) < 0:
        sys.stderr.write("WARNING: img2ndarr: iY has %d values < 0, clamping to 0\n" % np.sum(iY_flat<0))
        iY_flat[iY_flat < 0]=0
    if np.max(iX_flat) >= img.shape[0]:
        sys.stderr.write("WARNING: img2ndarr: iX has %d values >= img.shape[0]=%d, clamping to %d\n" % \
                         (np.sum(iX_flat >= img.shape[0]), img.shape[0], img.shape[0]-1))
        iX_flat[iX_flat >= img.shape[0]]=img.shape[0]-1
    if np.max(iY_flat) >= img.shape[1]:
        sys.stderr.write("WARNING: img2ndarr: iY has %d values >= img.shape[1]=%d, clamping to %d\n" % \
                         (np.sum(iY_flat >= img.shape[1]), img.shape[1], img.shape[1]-1))
        iY_flat[iY_flat >= img.shape[1]]=img.shape[1]-1
    ndarr[:] = img[iX_flat, iY_flat].reshape(ndarr.shape)
    return ndarr

__psana2DTypeLists=[psana.Epix.Element,
                    psana.Camera.Frame]

def typeIs2D(psanaType):
    for typeList in __psana2DTypeLists:
        if isinstance(typeList, list):
            if psanaType in typeList:
                return True
        elif isinstance(psanaType, typeList):
            return True
    return False

def getNdarr2ImageMapping(dsetstring, srcString, psanaType, psanaTypeStr, geom):
    if geom is None:
        try:
            geometryFile = getGeometryFile(dsetstring, srcString, psanaTypeStr)
        except NoGeomFile,e:
            if typeIs2D(psanaType):
                iX=None
                iY=None
                return iX, iY
            else:
                raise e
    else:
        assert os.path.exists(geom), "user supplied geometry file: %s not found" % geom
        geometryFile = geom

    geometry = GeometryAccess(geometryFile)
    iX, iY = geometry.get_pixel_coord_indexes()
    return iX, iY
    
def saveNdarr2ImgMapping(iX,iY,iXfname,iYfname):
    iXfout = file(iXfname, 'w')
    np.save(iXfout, iX)
    iXfout.close()

    iYfout = file(iYfname, 'w')
    np.save(iYfout, iY)
    iYfout.close()

def makeNdarrImageMappingFor2D(shape2D):
    iX = np.zeros(shape2D, np.int)
    iY = np.zeros(shape2D, np.int)
    for x in range(shape2D[0]):
        iX[x,:]=x
    for y in range(shape2D[1]):
        iY[:,y]=y
    return iX, iY

def turnOnTestPixels(arr,avg, numPixels=10, verbose=False):
    assert arr.shape == avg.shape
    sortedInds = np.argsort(avg.flatten())
    N = len(sortedInds)
    assert N>0
    ind0 = max(0,min(N-1,int(math.floor(.95*N))))
    ind1 = ind0 + numPixels
    if ind1 > N:
        ind0 = max(0,ind0-(ind1-N))
        ind1 = N
    ind0 = max(0,ind0)
    shapedIndicies = np.unravel_index(range(ind0,ind1),avg.shape)
    arr[shapedIndicies]=1
    if verbose:        
        for k in range(len(shapedIndicies[0])):
            inds=[indList[k] for indList in shapedIndicies]
            print "turnOnTestPixels: Turned on index: (%s)" % ','.join(map(str,inds))

def makeInitialFiles(dsetstring, psanaTypeStr, srcString, numForAverage=300, 
                     color=6, finecolor=18, basename=None, geom=None, debug=False, force=False,
                     numTestPixels=10, verboseForTestPixels=False):
    #### helper function ####
    def makeBaseName(dsetstring, srcString):
        '''make a base name from the dataset string and source
        '''
        basename = dsetstring
        basename = '_'.join(basename.split('exp='))
        basename = '-r'.join(basename.split('run='))
        basename += '_%s' % srcString
        okOrds = range(ord('a'),ord('z')+1) + range(ord('A'),ord('Z')+1) + \
                 range(ord('0'),ord('9')+1) + [ord('-'),ord('_')]
        def filter(x):
            if x in okOrds:
                return '%c'%x
            return '_'
        basename = ''.join(map(filter,map(ord,basename)))
        while basename.startswith('_'):
            basename = basename[1:]
        while basename.endswith('_'):
            basename = basename[0:-1]
        basename = basename.replace('_-','-')
        basename = basename.replace('_DetInfo_','_')
        print "INFO: generating basename=%s from dsetstring=%s and src=%s" % (basename, dsetstring, srcString)
        return basename
            
    #### start code ####
    if basename is not None:
        assert isinstance(basename,str) and len(basename)>0, "invalid basename for files"
    else:
        basename = makeBaseName(dsetstring, srcString)

    assert psanaTypeStr.startswith('psana.'), "psana type must start with psana."
    try:
        psanaType = eval(psanaTypeStr)
    except NameError,e:
        raise NameError("Unable to evaluate psana type from string: %s" % psanaTypeStr)
    except AttributeError,e:
        raise AttributeError("psana type string: % s has attribute that is wrong. Check type spec" % psanaTypeStr)

    iX, iY = getNdarr2ImageMapping(dsetstring, srcString, psanaType, psanaTypeStr, geom)

    iXfname = basename + '_iX.npy'
    iYfname = basename + '_iY.npy'

    avgNdarrFname = basename + '_avg_ndarrCoords.npy'
    avgImgFname = basename + '_avg_imageCoords.npy'

    maskNdarrFname = basename + '_mask_ndarrCoords.npy'
    maskImgFname = basename + '_mask_imageCoords.npy'

    testmaskNdarrFname = basename + '_testmask_ndarrCoords.npy'

    colorNdarrFname = basename + '_color_ndarrCoords.npy'
    colorImgFname = basename + '_color_imageCoords.npy'

    finecolorNdarrFname = basename + '_finecolor_ndarrCoords.npy'
    finecolorImgFname = basename + '_finecolor_imageCoords.npy'

    for fname in [iXfname, iYfname, avgNdarrFname, avgImgFname, 
                  maskNdarrFname, testmaskNdarrFname, maskImgFname, 
                  colorNdarrFname, colorImgFname, finecolorNdarrFname, finecolorImgFname]:
        assert (not os.path.exists(fname)) or force, "file %s exists. pass --force to overwrite" % fname

    if iX is not None and iY is not None:
        saveNdarr2ImgMapping(iX,iY,iXfname,iYfname)
        print "*** Analyzing ndarr -> img mapping from geometry file: ***"
        manyToOne = findImageManyToOne(iX,iY)
        if len(manyToOne)>0:
            print "INFO: There are %d image pixels with more than one ndarr pixel mapped to it" % len(manyToOne)
            for imagePos, ndarrList in manyToOne.iteritems():
                points = arrIndexListsToPoints(ndarrList)
                points = [','.join(map(str,point)) for point in points]
#                print "image pixel = %r  has ndarray elements = (%s)" % (imagePos, '), ('.join(points))
        else:
            print "INFO: all ndarr elements are mapped to a unique image pixel"
        imageGaps = findImageGaps(iX, iY)
        if len(imageGaps)>0:
            print "INFO: There appear to be %d gaps in the image, pixels in detctor areas for which no ndarr element is mapped" % \
                len(imageGaps)
            for gap in imageGaps:
                msg = '  gap:'
                for point in arrIndexListsToPoints(gap):
                    msg += ' (%s)' % ','.join(map(str,point))
 #               print msg
        print "*** done analyzing ndarr -> img mapping. Computing average. ***"

    # make a psana configuration to load ndarray producer and ndarrCalib to go through
    # events and make an average
    ndarray_out_key = "ndarray"
    calib_out_key = "calibrated"
    psanaOptions, outArrayType = PsanaUtil.makePsanaOptions(srcString, 
                                                            psanaType,
                                                            ndarray_out_key,
                                                            calib_out_key)
    
    psana.setOptions(psanaOptions)
    debugOut(debug,'psanaOptions:\n   %s\n' % '\n  '.join(map(lambda x: '%s=%s' % (x[0],x[1]),psanaOptions.iteritems())))
    ds = psana.DataSource(dsetstring)
    src = psana.Source(srcString)
    debugOut(debug, "src=%s\n" % src)

    ### psana event loop, get arrays to describe img <-> ndarray conversion, and get
    ### average of detector image

    eventsWithNoCalibNdarr = 0
    foundCalibNdarr = False

    ndarrAverage = None
    numInAverage = 1

    ### psana event loop
    for idx, evt in enumerate(ds.events()):
        calibNdarr = evt.get(outArrayType, src, calib_out_key)
        if calibNdarr is None: 
            eventsWithNoCalibNdarr += 1
            if eventsWithNoCalibNdarr > 130 and not foundCalibNdarr:
                break
            continue

        foundCalibNdarr = True
        if ndarrAverage is None:
            ndarrAverage = np.zeros(calibNdarr.shape, np.float64)
            ndarrAverage[:] = calibNdarr[:]
        else:
            numInAverage += 1
            ndarrAverage *= (float(numInAverage-1)/float(numInAverage))
            ndarrAverage[:] += ((1.0/float(numInAverage))*calibNdarr[:])
        if numInAverage > numForAverage:
            break
    ### end psana event loop

    if not foundCalibNdarr:
        msg = "ERROR: did not find a calibrated ndarray in %d events.\n" % eventsWithNoImage
        msg += "looking for psanaType=%s srcString=%s\n " % (psanaType, srcString)
        msg += "last event had: keys:\n  %s\n" % '\n  '.join(map(str,evt.keys()))
        raise Exception(msg)
        
    print "Formed ndarr average - saving."
    fout = file(avgNdarrFname,'w')
    np.save(fout, ndarrAverage)
    fout.close()

    if iX is None and iY is None:
        assert len(ndarrAverage.shape)==2, "Internal error - thought that ndarray producer for %s outputed a 2D array, but it did not. The shape is %r" % (psanaTypeStr, ndarrAverage.shape)
        iX, iY = makeNdarrImageMappingFor2D(ndarrAverage.shape)
        saveNdarr2ImgMapping(iX, iY, iXfname, iYfname)

    print "saving img average"
    fout = file(avgImgFname,'w')
    imgAverage = ndarr2img(ndarrAverage, iX, iY)
    np.save(fout, imgAverage)
    fout.close()

    maskNdarr = np.ones(ndarrAverage.shape,np.int8)
    print "saving a mask"
    fout = file(maskNdarrFname,'w')
    np.save(fout, maskNdarr)
    fout.close()

    testmaskNdarr = np.zeros(ndarrAverage.shape,np.int8)
    turnOnTestPixels(testmaskNdarr, ndarrAverage, numTestPixels, verboseForTestPixels)
    print "saving a testmask"
    fout = file(testmaskNdarrFname,'w')
    np.save(fout, testmaskNdarr)
    fout.close()

    print "saving img mask"
    fout = file(maskImgFname,'w')
    np.save(fout, ndarr2img(maskNdarr, iX, iY))
    fout.close()
    
    print "making ndarr color file and saving"
    ndarrColor = makeColorArray(ndarrAverage, color)
    fout = file(colorNdarrFname, 'w')
    np.save(fout, ndarrColor)
    fout.close()

    print "saving to img"
    fout = file(colorImgFname, 'w')
    np.save(fout, ndarr2img(ndarrColor, iX, iY))
    fout.close()
    
    print "making ndarr finecolor file with %d colors and saving" % finecolor
    finendarrColor = makeColorArray(ndarrAverage, finecolor)
    fout = file(finecolorNdarrFname, 'w')
    np.save(fout, finendarrColor)
    fout.close()

    print "saving to img"
    fout = file(finecolorImgFname, 'w')
    np.save(fout, ndarr2img(finendarrColor, iX, iY))
    fout.close()

def plotImageFile(inputFile):
    assert isinstance(inputFile,str) and len(inputFile)>0, "invalid input file"
    img = np.load(inputFile)
    imgMin = np.min(img)
    normImage = 1.0 + img - imgMin
    import matplotlib.pyplot as plt
    plt.ion()
    plt.imshow(np.log(normImage), interpolation='none')
    plt.draw()
    raw_input("hit enter to end")
