#--------------------------------------------------------------------------
# Description:
#   Test script for Translator/H5Output event processing
#   
#------------------------------------------------------------------------


#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os
import tempfile
import unittest
import subprocess as sb
import collections
import math
import numpy as np
#-----------------------------
# Imports for other modules --
#-----------------------------
import h5py

# -----------------------------
# Test data
# -----------------------------
DATADIR = "/reg/g/psdm/data_test/Translator"
OUTDIR = "data/Translator"
TESTDATA_T1= os.path.join(DATADIR, "test_042_Translator_t1.xtc")
TESTDATA_T1_INITIAL_DAMAGE = os.path.join(DATADIR,"test_046_Translator_t1_initial_damage.xtc")
TESTDATA_T1_END_DAMAGE = os.path.join(DATADIR,"test_045_Translator_t1_end_damage.xtc")
TESTDATA_T1_NEW_OUTOFORDER = os.path.join(DATADIR,"test_047_Translator_t1_new_out_of_order.xtc")
TESTDATA_T1_PREVIOUS_OUTOFORDER = os.path.join(DATADIR,"test_048_Translator_t1_previously_seen_out_of_order.xtc")
TESTDATA_AMO68413_r99_s2 = os.path.join(DATADIR,"test_041_Translator_amo68413-r99-s02-userEbeamDamage.xtc")
TESTDATA_AMO64913_r182_s2_OUTOFORDER_FRAME = os.path.join(DATADIR,'test_039_Translator_amo64913-r182-s02-OutOfOrder_Frame.xtc')
TESTDATA_AMO64913_r182_s2_NODAMAGE_DROPPED_OUTOFORDER = os.path.join(DATADIR,"test_040_Translator_amo64913-r182-s02-noDamage-dropped-OutOfOrder_Frame.xtc")
TESTDATA_XCSCOM12_r52_s0 = os.path.join(DATADIR,"test_049_Translator_xcscom12-r52-s0-dupTimes-splitEvents.xtc")
TESTDATA_T1_DROPPED_SRC = os.path.join(DATADIR,"test_044_Translator_t1_dropped_src.xtc")
TESTDATA_T1_DROPPED = os.path.join(DATADIR,"test_043_Translator_t1_dropped.xtc")
TESTDATA_ALIAS = os.path.join(DATADIR,"test_050_sxr_sxrb6813_e363-r0069-s00-c00.xtc")
TESTDATA_PARTITION = os.path.join(DATADIR,"test_051_sxr_sxrdaq10_e19-r057-s01-c00.xtc")
## ----------------
# this string is for the test_ndarrays_allWritenToFile test

NDARRAY_2EVENTS='''double3D:
array([[[[ 1.,  2.],
         [ 3.,  4.]],

        [[ 5.,  6.],
         [ 7.,  8.]]],


       [[[ 2.,  3.],
         [ 4.,  5.]],

        [[ 6.,  7.],
         [ 8.,  9.]]]])
float2Da:
array([[[ 1.,  2.],
        [ 3.,  4.]],

       [[ 2.,  3.],
        [ 4.,  5.]]], dtype=float32)
float2Db:
array([[[ 1.,  2.],
        [ 3.,  4.]],

       [[ 2.,  3.],
        [ 4.,  5.]]], dtype=float32)
int1D:
array([[1, 2],
       [2, 3]], dtype=int32)
str1:
array([This is event number: 1, This is event number: 2], dtype=object)
str2:
array([This is a second string.  10 * event number is 10,
       This is a second string.  10 * event number is 20], dtype=object)
'''

#------------------
# Utility functions 
#------------------
def makeH5OutputNameFromXtc(xtcfile):
    xtcbase = os.path.basename(xtcfile)
    h5out = os.path.join(OUTDIR, os.path.splitext(xtcbase)[0]) + '.h5'
    assert h5out != xtcfile, "xtcfile ends with .h5, it is %s" % xtcfile
    return h5out

def testDatasetsAgainstExpectedOutput(tester,h5,testList,cmpPlaces=4,verbose=False):
    '''asserts that the specified datasets within the h5 object have the specified values.
    ARGS:
    tester - an instance of unittest.testcase
    h5     - the h5 object, typically as returned by h5py.File
    testList - a list where each entry is as follows:
               (dataset_path, datset_values, message_if_test_fails)
    cmpPlaces - all comparisons are done with almostEquals, pass the number of decimal places
                to compare.
    verbose - set to True to get output for debugging purposes

    for example, if testList is [('/group/datasetA', [(0.1,0),(1.3,2)], 'first dataset'),
                                 ('/group/datasetB',[(0,1,2.1,3),(5,4,5.1,4)], 'second dataset')]
    this function will read h5['/group/datasetA'], test for two rows, and test that the
    first row is within 4 decimal places (or what is passed into cmpPlaces) of (0.1,0), and 
    the second is within 4 decimal places of (1.3,2).  It carries out a similar test for datasetB.
    '''
    for dsname,dsCmpTo,msg in testList:
        ds = h5[dsname]
        if (verbose):
            print "ds: %s" % dsname
        tester.assertEqual(len(ds),len(dsCmpTo), \
                         msg = "h5 dataset is wrong size.  ds.name=%s, len(ds)=%d len(dsCmpTo)=%d. %s" % (ds.name, len(ds), len(dsCmpTo), msg))
        for row in range(len(dsCmpTo)):
            if verbose:
                print "ds      row=%d: %r" % (row, ds[row])
                print "dsCmpTo row=%d: %r" % (row, dsCmpTo[row])
            if ds.dtype and ds.dtype.names:
                for i,fld in enumerate(ds.dtype.names):
                    if (verbose):
                        print "tester.assertAlmostEqual, dsCmpTo[%d][%d]=%r ds[%d][%s]=%r, places=%d" % \
                            (row,i,dsCmpTo[row][i],row,fld,ds[row][fld],cmpPlaces)
                    if fld == 'stamp':
                        tester.assertAlmostEqual(dsCmpTo[row][i][0],ds[row][fld][0],places=cmpPlaces, \
                                           msg="%s, dsname=%s, field is stamp.secsPastEpoch" % (msg,ds.name))
                        tester.assertAlmostEqual(dsCmpTo[row][i][1],ds[row][fld][1],places=cmpPlaces, \
                                           msg="%s, dsname=%s, field is stamp.nsec" % (msg,ds.name))
                    else:
                        tester.assertAlmostEqual(dsCmpTo[row][i],ds[row][fld],places=cmpPlaces, \
                                                 msg="%s, dsname=%s" % (msg,ds.name))
            else:
                if (verbose):
                    print "tester.assertAlmostEqual, dsCmpTo[%d]=%r ds[%d]=%r, places=%d" % \
                            (row,dsCmpTo[row],row,ds[row],cmpPlaces)
                tester.assertAlmostEqual(dsCmpTo[row],ds[row],places=cmpPlaces, \
                                       msg = "%s, dsname=%s" % (msg,ds.name))

                
def writeCfgFile(input_file, output_h5, moduleList="Translator.H5Output"):
    '''Starts to write a psana cfg file.  This is a temporary file.
    Returns the file like object so the user may add more options. This starts to 
    fill out the H5Output module options.
    '''
    cfgfile = tempfile.NamedTemporaryFile(suffix='.cfg',prefix='translator-unit-test')
    cfgfile.write("[psana]\n")
    cfgfile.write("modules = %s\n"%moduleList)
    cfgfile.write("files = %s\n" % input_file)
    cfgfile.write("[Translator.H5Output]\n")
    cfgfile.write("output_file = %s\n" % output_h5)
    cfgfile.write("Epics=exclude\n")  # to exclude the epics::ConfigV1, to get things to look 
                                      # more like o2o-translate
    cfgfile.file.flush();
    return cfgfile


#-------------------------------
#  Unit test class definition --
#-------------------------------
class H5Output( unittest.TestCase ) :

    def setUp(self) :
    	""" 
    	Method called to prepare the test fixture. This is called immediately 
    	before calling the test method; any exception raised by this method 
    	will be considered an error rather than a test failure.  
    	"""
        assert os.path.exists(DATADIR), "Data dir: %s does not exist, cannot run unit tests" % DATADIR
        assert os.path.exists(OUTDIR), "Output directory: %s does not exist, can't run unit tests" % OUTDIR
        self.cleanUp = True      # Several tests run psana to produce .h5 files.  
                                 # If cleanup is True the files are deleted when
                                 # the test is done.

        self.printPsanaOutput = False # if True, when a test psana it will write
                                      # it's output.  It will also call h5ls -r on
                                      # on the h5 file created.
    def tearDown(self) :
        """
        Method called immediately after the test method has been called and 
        the result recorded. This is called even if the test method raised 
        an exception, so the implementation in subclasses may need to be 
        particularly careful about checking internal state. Any exception raised 
        by this method will be considered an error rather than a test failure. 
        This method will only be called if the setUp() succeeds, regardless 
        of the outcome of the test method. 
        """
        pass

    def runPsanaOnCfg(self,cfgfile,output_h5,extraOpts='',printPsanaOutput=False, errorCheck=True):
        '''Runs psana on the given cfgfile, to produce the given h5output file.
        extraOpts are passed to psana on the command line.
        
        tests that output_h5 is created.

        If errorCheck is True it tests that psana output does not include: fatal, error,
                        segmentation fault, seg falut, traceback

        returns the output of psana
        '''
        if os.path.exists(output_h5):
            os.unlink(output_h5)
        cfgfile.flush()
        assert isinstance(extraOpts,str), "extraOpts for psana command line is %r, not a str" % extraOpts
        psana_cmd = "psana %s -c %s" % (extraOpts,cfgfile.name)
        p = sb.Popen(psana_cmd,shell=True,stdout=sb.PIPE, stderr=sb.PIPE)
        o,e = p.communicate()
        if not os.path.exists(output_h5):
            print "### h5 file was not created. cfg file: ###"
            print file(cfgfile.name).read()
            print "### psana output: ###"
            print o
            print e
        self.assertEqual(os.path.exists(output_h5),True)
        allOutPut = o+'\n'+e
        if printPsanaOutput:
            print allOutPut
            sys.stdout.flush()
            print "*** Running h5ls -r on output_h5 (%s)"% output_h5
            os.system('h5ls -r %s | grep -v -i epics' % output_h5)
        if not errorCheck:
            return allOutPut
        lowerOutput = allOutPut.lower()
        self.assertEqual(lowerOutput.find('fatal'),-1,msg="'fatal' found in psana output: ... %s ..." % lowerOutput[lowerOutput.find('fatal')-100:lowerOutput.find('fatal')+100])
        self.assertEqual(lowerOutput.find('error'),-1,msg="'error' found in psana output: ... %s ..."  % lowerOutput[lowerOutput.find('error')-100:lowerOutput.find('error')+100])
        self.assertEqual(lowerOutput.find('segmentation fault'),-1,msg="'segmentation fault' found in psana output: ... %s ..." % lowerOutput[lowerOutput.find('segmentation fault')-100:lowerOutput.find('segmentation fault')+100])
        self.assertEqual(lowerOutput.find('seg fault'),-1,msg="'seg fault' found in psana output: ... %s ..." % lowerOutput[lowerOutput.find('seg fault')-100:lowerOutput.find('seg fault')+100])
        self.assertEqual(lowerOutput.find('traceback'),-1,msg="'traceback' found in psana output: ... %s ..." % lowerOutput[lowerOutput.find('traceback')-100:lowerOutput.find('traceback')+100])

    def test_t1_initial_damage(self):
        '''check for initial blanks.
        The input file is a modified version of t1.xtc.
        Damage has been introduced for the first of the two Ipimb::Data types
        coming from the XppSb3_Ipm source.  Hence we should get a blank starting
        that dataset.  This will test the initial_blank logic of 
        H5Output::Event and TypeSrcKeyDirectory.
        '''
        input_file = TESTDATA_T1_INITIAL_DAMAGE
        output_h5 = makeH5OutputNameFromXtc(input_file)
        cfgfile = writeCfgFile(input_file, output_h5)
        self.runPsanaOnCfg(cfgfile,output_h5, printPsanaOutput=self.printPsanaOutput)
        cfgfile.close()
        h5 = h5py.File(output_h5,'r')

        eventIds = [(1364147551, 107587445, 331318, 118410, 140, 0), 
                    (1364147551, 174323092, 331570, 118434, 12, 6)]
        blankDamage = [(4, 0, 0, 0, 0, 0, 0),
                       (0, 0, 0, 0, 0, 0, 0)]
        nonBlankDamage = [(0, 0, 0, 0, 0, 0, 0),
                          (0, 0, 0, 0, 0, 0, 0)]
        blankMask = [0,1]
        nonBlankMask = [1,1]
        blankData = [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 
                    (4369, 0, 31250, 65486, 65535, 65324, 65447, 42142, 41839, 42184, 42083, 42042, 4.99626, 5, 4.9839, 4.99329, 3.21523, 3.19211, 3.21843, 3.21073, 18135826)]
        nonBlankData = [(4369, 0, 31250, 65525, 65522, 65517, 65532, 41975, 41984, 41949, 41903, 44883, 4.99924, 4.99901, 4.99863, 4.99977, 3.20249, 3.20317, 3.2005, 3.19699, 18135819),
                        (4369, 0, 31250, 65462, 65483, 65523, 65508, 42069, 42050, 41978, 41927, 47071, 4.99443, 4.99603, 4.99908, 4.99794, 3.20966, 3.20821, 3.20272, 3.19883, 18135827)]

        testList = [('Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb3_Ipm/data', blankData, "data dataset with blank"),
                    ('Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb3_Ipm/_damage', blankDamage, "damage dataset with blank"),
                    ('Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb3_Ipm/time', eventIds, "time dataset with blank"),
                    ('Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb3_Ipm/_mask', blankMask, "mask dataset with blank"),
                    ('Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb2_Ipm/data', nonBlankData, "data dataset with non blank"),
                    ('Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb2_Ipm/_damage', nonBlankDamage, "damage dataset with non blank"),
                    ('Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb2_Ipm/time', eventIds, "time dataset with non blank"),
                    ('Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb2_Ipm/_mask', nonBlankMask, "mask dataset with non blank")]
        testDatasetsAgainstExpectedOutput(self,h5,testList,cmpPlaces=4)

        if self.cleanUp:
            os.unlink(output_h5)

    def test_t1_end_damage(self):
        '''Check for blanks at the end of the datasets.
        The input file is a modified version of t1.xtc.
        Damage has been introduced at the end of the Ipimb::Data types
        coming from the XppSb3_Ipm source.  Hence we should get a blank at the end
        of that dataset. 
        '''
        input_file = TESTDATA_T1_END_DAMAGE
        output_h5 = makeH5OutputNameFromXtc(input_file)
                         
        cfgfile = writeCfgFile(input_file, output_h5)
        self.runPsanaOnCfg(cfgfile,output_h5, printPsanaOutput=self.printPsanaOutput)
        cfgfile.close()
        h5 = h5py.File(output_h5,'r')

        eventIds = [(1364147551, 107587445, 331318, 118410, 140, 0), 
                    (1364147551, 174323092, 331570, 118434, 12, 6)]
        blankDamage = [(0, 0, 0, 0, 0, 0, 0),
                       (4, 0, 0, 0, 0, 0, 0)]
        nonBlankDamage = [(0, 0, 0, 0, 0, 0, 0),
                          (0, 0, 0, 0, 0, 0, 0)]
        blankMask = [1,0]
        nonBlankMask = [1,1]
        blankData = [(4369, 0, 31250, 65535, 65535, 65535, 65535, 42105, 41813, 42024, 42019, 26799, 5, 5, 5, 5, 3.21241, 3.19013, 3.20623, 3.20584, 18135818),
                     (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
        nonBlankData = [(4369, 0, 31250, 65525, 65522, 65517, 65532, 41975, 41984, 41949, 41903, 44883, 4.99924, 4.99901, 4.99863, 4.99977, 3.20249, 3.20317, 3.2005, 3.19699, 18135819),
                        (4369, 0, 31250, 65462, 65483, 65523, 65508, 42069, 42050, 41978, 41927, 47071, 4.99443, 4.99603, 4.99908, 4.99794, 3.20966, 3.20821, 3.20272, 3.19883, 18135827)]

        testList = [('Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb3_Ipm/data', blankData, "data dataset with blank"),
                    ('Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb3_Ipm/_damage', blankDamage, "damage dataset with blank"),
                    ('Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb3_Ipm/time', eventIds, "time dataset with blank"),
                    ('Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb3_Ipm/_mask', blankMask, "mask dataset with blank"),
                    ('Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb2_Ipm/data', nonBlankData, "data dataset with non blank"),
                    ('Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb2_Ipm/_damage', nonBlankDamage, "damage dataset with non blank"),
                    ('Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb2_Ipm/time', eventIds, "time dataset with non blank"),
                    ('Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb2_Ipm/_mask', nonBlankMask, "mask dataset with non blank")]
        
        testDatasetsAgainstExpectedOutput(self,h5,testList,cmpPlaces=4)

        if self.cleanUp:
            os.unlink(output_h5)

    def test_t1_new_outoforder(self):
        '''Check for proper handling of outoforder damage.
        The input file is a modified version of t1.xtc. If has out of order damage for the
        second event for ipimb data from sb32.
        '''
        input_file = TESTDATA_T1_NEW_OUTOFORDER
        output_h5 = makeH5OutputNameFromXtc(input_file)
        cfgfile = writeCfgFile(input_file, output_h5)
        self.runPsanaOnCfg(cfgfile,output_h5, printPsanaOutput=self.printPsanaOutput)
        cfgfile.close()
        h5 = h5py.File(output_h5,'r')
        eventIds = [(1364147551, 107587445, 331318, 118410, 140, 0), 
                    (1364147551, 174323092, 331570, 118434, 12, 6)]
        blankDamage = [(0, 0, 0, 0, 0, 0, 0),
                       (4096, 0, 1, 0, 0, 0, 0)]
        nonBlankDamage = [(0, 0, 0, 0, 0, 0, 0),
                          (0, 0, 0, 0, 0, 0, 0)]
        blankMask = [1,0]
        nonBlankMask = [1,1]
        nonBlankData = [(4369, 0, 31250, 65535, 65535, 65535, 65535, 42105, 41813, 42024, 42019, 26799, 5, 5, 5, 5, 3.21241, 3.19013, 3.20623, 3.20584, 18135818),
                        (4369, 0, 31250, 65486, 65535, 65324, 65447, 42142, 41839, 42184, 42083, 42042, 4.99626, 5, 4.9839, 4.99329, 3.21523, 3.19211, 3.21843, 3.21073, 18135826) ]
        blankData = [(4369, 0, 31250, 65525, 65522, 65517, 65532, 41975, 41984, 41949, 41903, 44883, 4.99924, 4.99901, 4.99863, 4.99977, 3.20249, 3.20317, 3.2005, 3.19699, 18135819),
                     (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]

        testList = [('Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb2_Ipm/data', blankData, "data dataset with blank"),
                    ('Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb2_Ipm/_damage', blankDamage, "damage dataset with blank"),
                    ('Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb2_Ipm/time', eventIds, "time dataset with blank"),
                    ('Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb2_Ipm/_mask', blankMask, "mask dataset with blank"),
                    ('Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb3_Ipm/data', nonBlankData, "data dataset with non blank"),
                    ('Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb3_Ipm/_damage', nonBlankDamage, "damage dataset with non blank"),
                    ('Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb3_Ipm/time', eventIds, "time dataset with non blank"),
                    ('Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb3_Ipm/_mask', nonBlankMask, "mask dataset with non blank")]
        
        testDatasetsAgainstExpectedOutput(self,h5,testList,cmpPlaces=4)
        if self.cleanUp:
            os.unlink(output_h5)

    def test_t1_previously_seen_out_of_order(self):
        '''check for proper damage handling.
        The input file is a modified version of t1.xtc. There are two L1Accept datagrams with the
        same time.  The first one has damage, and the second one says out of order.

        The datagrams:

        dg=    5 offset=0x0000FE98 tp=Event sv=       L1Accept ex=1 ev=1 sec=514F3D5F nano=0669A775 tcks=0050E36 fid=1CE8A ctrl=8C vec=0000 env=00000003
        dg=    6 offset=0x0001327C tp=Event sv=       L1Accept ex=0 ev=1 sec=514F3D5F nano=0669A775 tcks=0050F32 fid=1CEA2 ctrl=0C vec=0006 env=00000003
        
        The ipimb data is dg5:
        offset=0FED4 extent=034 dmg=00004 src=06003D77,00000025,level=6 typeid=22 vrn=2 val=20016 type_name=IpimbData
        offset=0FFD4 extent=034 dmg=00005 src=06003D77,00000024,level=6 typeid=22 vrn=2 val=20016 type_name=IpimbData
        
        In dg6:
        offset=132F0 extent=034 dmg=01000 src=06003D77,00000025,level=6 typeid=22 vrn=2 val=20016 type_name=IpimbData
        offset=133F0 extent=034 dmg=01000 src=06003D77,00000024,level=6 typeid=22 vrn=2 val=20016 type_name=IpimbData

        psana appears to sort the datagrams so that we get dg6 before dg5.

        Since every entry is blank, the psana Translator never gets a chance to write the blanks.
        What it produces is the time, _damage and _mask datasets, but it does not write the
        data datasets.
        '''
        input_file = TESTDATA_T1_PREVIOUS_OUTOFORDER
        output_h5 = makeH5OutputNameFromXtc(input_file)
        cfgfile = writeCfgFile(input_file, output_h5)
        self.runPsanaOnCfg(cfgfile,output_h5, printPsanaOutput=self.printPsanaOutput)
        cfgfile.close()
        h5 = h5py.File(output_h5,'r')
        sec = 1364147551
        nano = 107587445
        timeSb2 = h5['/Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb2_Ipm/time']
        timeSb3 = h5['/Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb3_Ipm/time']
        self.assertEqual(len(timeSb2),2,"should be two entries in ipimb data time")
        self.assertEqual(len(timeSb3),2,"should be two entries in ipimb data time")
        self.assertEqual(timeSb2['seconds'][0],sec)
        self.assertEqual(timeSb3['seconds'][0],sec)
        self.assertEqual(timeSb2['nanoseconds'][0],nano)
        self.assertEqual(timeSb3['nanoseconds'][0],nano)
        maskSb2 = h5['/Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb2_Ipm/_mask']
        maskSb3 = h5['/Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb3_Ipm/_mask']
        self.assertEqual(len(maskSb2),2,"should be two entries in ipimb data mask")
        self.assertEqual(len(maskSb3),2,"should be two entries in ipimb data mask")
        self.assertTrue((maskSb2[...]==np.zeros(2)).all())
        self.assertTrue((maskSb3[...]==np.zeros(2)).all())
        self.assertFalse('data' in h5['/Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb2_Ipm'].keys(),msg="should not have a 'data' dataset for ipimb")
        self.assertFalse('data' in h5['/Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2/XppSb3_Ipm'].keys(),msg="should not have a 'data' dataset for ipimb")
        if self.cleanUp:
            os.unlink(output_h5)

    def test_userEBeamDamage(self):
        '''Runs the translator on a file that has EBeam user defined damage. 
        The input file, data/Translator/amo68413-r99-s02-userEbeamDamage.xtc,
        was obtained by taking a mix of datagrams from different chunks of 
        amo68413 r0099 s02.  
        The only damage in the file is dgram 5, and L1Accept, with 
        xtc d=2  offset=0x00033854 extent=00000068 dmg=04000 src=06000000,00000000,level=6 typeid=15 version=3 value=3000F compressed=0 compressed_version=3 type_name=EBeamBld
        The 
        xdmg=04000The first 5 datagrams from cover the damage.  The next two get the end
        of the calibcyle, and then we go into chunk 03 to get the end of the run.

        That is we put the stared datagrams from the list below, s is the sequence,
        c the chunk, dg the datagram number (1-up).
        
        * s=2 c=0 dg=    1 offset=0x00000000 tp=Event sv=      Configure ex=1 ev=0 sec=510EE016 nano=069CA483 tcks=0000000 fid=1FFFF ctrl=84 vec=0000 env=000007A2
        * s=2 c=0 dg=    2 offset=0x0000C2F0 tp=Event sv=       BeginRun ex=0 ev=0 sec=510EE332 nano=16D0CC0D tcks=0000000 fid=1FFFF ctrl=06 vec=0000 env=00000063
        * s=2 c=0 dg=    3 offset=0x0000C37C tp=Event sv=BeginCalibCycle ex=0 ev=0 sec=510EE332 nano=217CB9C0 tcks=0000000 fid=1FFFF ctrl=08 vec=0000 env=00000000
        * s=2 c=0 dg=    4 offset=0x0000C5D4 tp=Event sv=         Enable ex=0 ev=0 sec=510EE332 nano=2510C571 tcks=0000000 fid=1FFFF ctrl=0A vec=0000 env=800007D0
        * s=2 c=0 dg=    5 offset=0x0000C660 tp=Event sv=       L1Accept ex=1 ev=1 sec=510EE332 nano=2A2C740E tcks=00508A0 fid=11670 ctrl=8C vec=40CD env=00000003
          s=2 c=0 dg=    6 offset=0x00233A74 tp=Event sv=       L1Accept ex=1 ev=1 sec=510EE332 nano=2AAB8F9C tcks=005077A fid=11673 ctrl=8C vec=40CE env=00000003
        * s=2 c=0 dg= 2008 offset=0x10C853038 tp=Event sv=        Disable ex=0 ev=0 sec=510EE343 nano=29E1CCFD tcks=0000000 fid=1FFFF ctrl=0B vec=0000 env=00000000
        * s=2 c=0 dg= 2009 offset=0x10C8530C4 tp=Event sv=  EndCalibCycle ex=0 ev=0 sec=510EE343 nano=2C442175 tcks=0000000 fid=1FFFF ctrl=09 vec=0000 env=00000000
          s=2 c=0 dg= 2010 offset=0x10C853150 tp=Event sv=BeginCalibCycle ex=0 ev=0 sec=510EE343 nano=36576935 tcks=0000000 fid=1FFFF ctrl=08 vec=0000 env=00000000
          s0  c=3 dg=16053 offset=0x864011A14 tp=Event sv=       L1Accept ex=1 ev=1 sec=510EE5F9 nano=0115D1F4 tcks=0050F3E fid=0FDBC ctrl=8C vec=0197 env=00000003
        * s0  c=3 dg=16054 offset=0x864236B60 tp=Event sv=        Disable ex=0 ev=0 sec=510EE5F9 nano=14211335 tcks=0000000 fid=1FFFF ctrl=0B vec=0000 env=00000000
        * s0  c=3 dg=16055 offset=0x864236BEC tp=Event sv=  EndCalibCycle ex=0 ev=0 sec=510EE5F9 nano=171C0FDA tcks=0000000 fid=1FFFF ctrl=09 vec=0000 env=00000000
        * s0  c=3 dg=16056 offset=0x864236C78 tp=Event sv=         EndRun ex=0 ev=0 sec=510EE5F9 nano=1A16F021 tcks=0000000 fid=1FFFF ctrl=07 vec=0000 env=00000000

        This test revealed some differences in epics.  First o2o-translate records the
        epics source, and they come from two places in this xtc file:  
           EpicsArch.0:NoDevice.0.  and AmoVMI.0:Opal1000.0
        Translator.H5Output puts everything in EpicsArch.0:NoDevice.0
        We also find that there are several aliases that o2o writes that psana does not:
        
        piezo timing delay _ width-AMO:R14:EVR:21:CTRL.DG2W              we skipped because it has an empty target
        mirror bendersreg_g_pcds_package_epics_3.14-dev_screens_edm_amo  we skipped because it has an empty target
        piezo timing delay _ width                                       we skipped because it has an empty target
        AMO:R14:IOC:10:ao0:out1                                          we skipped because it was in the group list
        
        This file revealed a bug in o2o-translate, it was creating a group for AMO:R14:IOC:10:ao0:out1 that is
        wrong.

        In this test, we'll make sure AMO:R14:IOC:10:ao0:out1 has the correct data, as well as that 
        the user damaged ebeam data is present. psana has a switch to not store damaged user ebeam, but
        by default it is on, so it should be stored.
        '''
        input_file = TESTDATA_AMO68413_r99_s2
        output_h5 = makeH5OutputNameFromXtc(input_file)
        cfgfile = writeCfgFile(input_file,output_h5)
        self.runPsanaOnCfg(cfgfile,output_h5)
        cfgfile.close()
        f = h5py.File(output_h5,'r')
        testList = [('/Configure:0000/Run:0000/CalibCycle:0000/Bld::BldDataEBeamV3/EBeam/_damage',
                     [(0x4000,0,0,0,1,0,0)], 'ebeam damage'),
                    ('/Configure:0000/Run:0000/CalibCycle:0000/Bld::BldDataEBeamV3/EBeam/_mask',
                     [(1,)], 'ebeam mask'),
                    ('/Configure:0000/Run:0000/CalibCycle:0000/Bld::BldDataEBeamV3/EBeam/data',
                     [(256L, 0.019605993696, 4813.759084237167, -0.21048433685050208, 0.008341444963823323, 0.032870595392491646, 0.0634931790398521, 15285.3740234375, 0.4098030626773834, 56.18963623046875, -0.2480001449584961)],
                     'ebeam data'),
                    ('/Configure:0000/Epics::EpicsPv/EpicsArch.0:NoDevice.0/AMO:R14:IOC:10:VHS0:CH0:VoltageMeasure/data',
                     [ (44, 34, 1, 'AMO:R14:IOC:10:VHS0:CH0:VoltageMeasure', 5, 2, 3, 'V', 5000.0, 0.0, 5000.0, 5000.0, 0.0, 0.0, 5000.0, 0.0, 0.0)],
                     'config epics amo:r14:0c:10:vhs0:ch0:voltage measure'),
                    ('/Configure:0000/Run:0000/CalibCycle:0000/Epics::EpicsPv/EpicsArch.0:NoDevice.0/AMO:R14:IOC:10:VHS0:CH0:VoltageMeasure/data',
                     [(44, 20, 1, 5, 2, (728768506L, 940000000L), 0.0)],
                     'config epics amo:r14:0c:10:vhs0:ch0:voltage measure')]

        testDatasetsAgainstExpectedOutput(self,f,testList)
        if self.cleanUp:
            os.unlink(output_h5)

    def test_outOfOrderFrame(self):
        '''Test proper damage handling. This file has
        dg1=config, dg2=beginRun, dg3=beginCalib, dg4=Enable
        dg5=L1Accept with:  sec=5159D9BB nano=37A715E2
          xtc extent=00000014 dmg=00002 src=01003A03,17010300,level=1 typeid= 0 version=0 value=00000 type_name=Any
        dg6=L1Accept with:  sec=5159D9BB nano=38A560D4
          xtc extent=00200024 dmg=01000 src=01003A03,17010300,level=1 typeid= 2 version=1 value=10002 type_name=Frame compressed=0 compressed_version=1 
        dg7=L1Accept with   sec=5159D9BB nano=39A38A8B 
          xtc extent=00200024 dmg=01000 src=01003A03,17010300,level=1 typeid= 2 version=1 value=10002 type_name=Frame compressed=0 compressed_version=1 

        Although we could keep track of the initial damage to an Any type at the given src, we presently do not.  
        initial damage to known types we do keep track of, but here we only know the src. When we see damage
        for sources where we have already seen data, we try to do more. So this data will
        produce a mis-aligned dataset with only two entries (both damaged).  Moreover, we will not 
        write the data dataset since we never see valid data for Frame.  

        If we find this kind of damage - src only in the inital datagrams - is happening, we could
        work on aligning this kind of dataset.
        '''
        input_file = TESTDATA_AMO64913_r182_s2_OUTOFORDER_FRAME
        output_h5 = makeH5OutputNameFromXtc(input_file)
        cfgfile = writeCfgFile(input_file, output_h5)
        self.runPsanaOnCfg(cfgfile,output_h5, printPsanaOutput=self.printPsanaOutput)
        cfgfile.close()
        h5 = h5py.File(output_h5,'r')
        gr = h5["/Configure:0000/Run:0000/CalibCycle:0000/Camera::FrameV1/AmoEndstation.1:Opal1000.0"]
        self.assertEqual(set(gr.keys()),set(['time','_damage','_mask']))
        frameDamage = gr["_damage"]
        frameMask = gr["_mask"]
        self.assertEqual(len(frameDamage),2)
        self.assertEqual(len(frameMask),2)
        self.assertTrue((frameMask[...]==np.zeros(2)).all())
        self.assertEqual(frameDamage['bits'][0],4096)
        self.assertEqual(frameDamage['bits'][1],4096)
        self.assertEqual(frameDamage['OutOfOrder'][0],1)
        self.assertEqual(frameDamage['OutOfOrder'][1],1)
        if self.cleanUp:
            os.unlink(output_h5)

    def test_noDamageDroppedOutOfOrder(self):
        '''Check proper damage handling. This file has
        dg1=config, dg2=beginRun, dg3=beginCalib, dg4=Enable
        dg5=L1Accept with:  sec=5159D9BB nano=36A8E830
          xtc extent=00200024 dmg=00000 src=01003A03,17010300, typeid= 2 type_name=Frame
        dg6=L1Accept with:  sec=5159D9BB nano=37A715E2 
          xtc extent=00000014 dmg=00002 src=01003A03,17010300, typeid= 0 type_name=Any
        dg7=L1Accept with:  sec=5159D9BB nano=38A560D4 
          xtc extent=00200024 dmg=01000 src=01003A03,17010300, typeid= 2 type_name=Frame
        dg8=L1Accept with:  sec=5159D9BB nano=39A38A8B
          xtc extent=00200024 dmg=01000 src=01003A03,17010300, typeid= 2 type_name=Frame

        Since we have seen the data from the dropped src with the Any xtc, we sould put a 
        blank in there.  So we should get a dataset of 4 here, with a mask of 1 0 0 0 
        '''
        input_file = TESTDATA_AMO64913_r182_s2_NODAMAGE_DROPPED_OUTOFORDER
        output_h5 = makeH5OutputNameFromXtc(input_file)
        cfgfile = writeCfgFile(input_file, output_h5)
        self.runPsanaOnCfg(cfgfile,output_h5, printPsanaOutput=self.printPsanaOutput)
        cfgfile.close()
        h5 = h5py.File(output_h5,'r')
        gr = h5["/Configure:0000/Run:0000/CalibCycle:0000/Camera::FrameV1/AmoEndstation.1:Opal1000.0"]
        self.assertEqual(set(gr.keys()),set(['time','_damage','_mask', 'data','image']))
        damage = gr["_damage"]
        mask = gr["_mask"]
        self.assertEqual(len(damage),4)
        self.assertEqual(len(mask),4)
        self.assertTrue((mask[...]==np.array([1,0,0,0])).all())
        if self.cleanUp:
            os.unlink(output_h5)
        
    def test_dupTimesSplitEvents(self):
        '''Check that we run without on error on file with split events.
        TODO: check for proper handling of split events.
        This file contains several pairs of datagrams with the same timestamps:
        dg=    8 sec=4F540ABB nano=0A5FC451  xtc dmg=00002 src=02001038,AC151974, type_name=Any
        dg=   11 sec=4F540ABB nano=0A5FC451  all xtc dmg=00002

        dg=    9 sec=4F540ABB nano=0D5B5A93 xtc dmg=00002 src=02001038,AC151974, type_name=Any
        dg=   14 sec=4F540ABB nano=0D5B5A93 all xtc dmg=00002

        dg=   10 sec=4F540ABB nano=1056D6A7 xtc dmg=00002 src=02001038,AC151974, type_name=Any
        dg=   15 sec=4F540ABB nano=1056D6A7 all xtc dmg=00002

        dg=   12 sec=4F540ABB nano=1352778B xtc dmg=00002 src=02001038,AC151974, type_name=Any
        dg=   16 sec=4F540ABB nano=1352778B all xtc dmg=00002

        dg=   13 sec=4F540ABB nano=164DEBBF xtc dmg=00002 src=02001038,AC151974, type_name=Any
        dg=   17 sec=4F540ABB nano=164DEBBF all xtc, smg=00002
        '''
        input_file = TESTDATA_XCSCOM12_r52_s0
        output_h5 = makeH5OutputNameFromXtc(input_file)
        cfgfile = writeCfgFile(input_file, output_h5)
        self.runPsanaOnCfg(cfgfile,output_h5, printPsanaOutput=self.printPsanaOutput)
        cfgfile.close()
        if self.cleanUp:
            os.unlink(output_h5)

    def test_t1_dropped_src(self):
        '''Check that we run successful on this xtc file with damage.
        '''
        input_file = TESTDATA_T1_DROPPED_SRC
        output_h5 = makeH5OutputNameFromXtc(input_file)
        cfgfile = writeCfgFile(input_file, output_h5)
        self.runPsanaOnCfg(cfgfile,output_h5, printPsanaOutput=self.printPsanaOutput)
        cfgfile.close()
        if self.cleanUp:
            os.unlink(output_h5)
        
    def test_t1_dropped(self):
        '''Check that we run successful on this xtc file with damage.
        '''
        input_file = TESTDATA_T1_DROPPED
        output_h5 = makeH5OutputNameFromXtc(input_file)
        cfgfile = writeCfgFile(input_file, output_h5)
        self.runPsanaOnCfg(cfgfile,output_h5, printPsanaOutput=self.printPsanaOutput)
        cfgfile.close()
        if self.cleanUp:
            os.unlink(output_h5)

    def test_doNotTranslate_addKey(self):
        '''Test doNotTranslate with a key string like "mytest:do_not_translate"
        '''
        input_file = TESTDATA_T1
        output_h5 = os.path.join(OUTDIR,'unit-test_doNotTranslate_addKey.h5')
        cfgfile = writeCfgFile(input_file,output_h5,"Translator.TestModuleDoNotTranslate Translator.H5Output")
        cfgfile.write("[Translator.TestModuleDoNotTranslate]\n")
        cfgfile.write("skip=0 1\n")
        msg0='message0'
        msg1='message1thisIsALongerMessage'
        cfgfile.write("messages=%s %s\n"% (msg0, msg1))
        cfgfile.write("key=mytest\n");
        cfgfile.file.flush()
        try:
            self.runPsanaOnCfg(cfgfile,output_h5)
        except:
            cfgfile.close()
            if os.path.exists(output_h5):
                os.unlink(output_h5)
            raise
        f=h5py.File(output_h5,'r')
        # make sure event data is not present:
        with self.assertRaises(KeyError):
            f['/Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2']
        with self.assertRaises(KeyError):
            f['/Configure:0000/Run:0000/CalibCycle:0000/EvrData::DataV3']
            
        # make sure data for filtered group is present
        filteredEventIds = f['/Configure:0000/Run:0000/Filtered:0000/time']
        self.assertTrue(len(filteredEventIds)==2 and
                        filteredEventIds['seconds'][0]==1364147551 and
                         filteredEventIds['seconds'][1]==1364147551 and
                         filteredEventIds['nanoseconds'][0]==107587445 and
                         filteredEventIds['nanoseconds'][1]==174323092, 
                        msg="time dataset not right in Filtered:0000/time")
        filteredMsgs = map(str,f['/Configure:0000/Run:0000/Filtered:0000/std::string/mytest/data'])
        self.assertEqual(filteredMsgs[0],msg0)
        self.assertEqual(filteredMsgs[1],msg1)
        filteredMsgsEventIds = f['/Configure:0000/Run:0000/Filtered:0000/std::string/mytest/time']
        self.assertTrue(len(filteredMsgsEventIds)==2 and
                        filteredMsgsEventIds['seconds'][0]==1364147551 and
                         filteredMsgsEventIds['seconds'][1]==1364147551 and
                         filteredMsgsEventIds['nanoseconds'][0]==107587445 and
                         filteredMsgsEventIds['nanoseconds'][1]==174323092, 
                        msg="time dataset not right in Filtered:0000/std::string/mytest/time")
        # we don't write damage to the filtered groups:
        with self.assertRaises(KeyError):
            f['/Configure:0000/Run:0000/Filtered:0000/std::string/mytest/_damage']
        with self.assertRaises(KeyError):
            f['/Configure:0000/Run:0000/Filtered:0000/std::string/mytest/_mask']

        if self.cleanUp:
            os.unlink(output_h5)
        
    def test_doNotTranslate(self):
        '''Test doNotTranslate with a basic key string: "do_not_translate"
        '''
        input_file = TESTDATA_T1
        output_h5 = os.path.join(OUTDIR,'unit_test_doNotTranslate.h5')
        cfgfile = writeCfgFile(input_file,output_h5,"Translator.TestModuleDoNotTranslate Translator.H5Output")
        cfgfile.write("[Translator.TestModuleDoNotTranslate]\n")
        cfgfile.write("skip=0 1\n")
        msg0='message0'
        msg1='message1thisIsALongerMessage'
        cfgfile.write("messages=%s %s\n"% (msg0, msg1))
        cfgfile.file.flush()
        try:
            self.runPsanaOnCfg(cfgfile,output_h5)
        except:
            cfgfile.close()
            if os.path.exists(output_h5):
                os.unlink(output_h5)
            raise
        f=h5py.File(output_h5,'r')
        
        # make sure event data is not present:
        with self.assertRaises(KeyError):
            f['/Configure:0000/Run:0000/CalibCycle:0000/Ipimb::DataV2']
        with self.assertRaises(KeyError):
            f['/Configure:0000/Run:0000/CalibCycle:0000/EvrData::DataV3']
            
        # make sure filtered data is present
        filteredEventIds = f['/Configure:0000/Run:0000/Filtered:0000/time']
        self.assertTrue(len(filteredEventIds)==2 and
                        filteredEventIds['seconds'][0]==1364147551 and
                         filteredEventIds['seconds'][1]==1364147551 and
                         filteredEventIds['nanoseconds'][0]==107587445 and
                         filteredEventIds['nanoseconds'][1]==174323092, 
                        msg="time dataset not right in Filtered:0000/time")
        filteredMsgs = map(str,f['/Configure:0000/Run:0000/Filtered:0000/std::string/no_src/data'])
        self.assertEqual(filteredMsgs[0],msg0)
        self.assertEqual(filteredMsgs[1],msg1)
        filteredMsgsEventIds = f['/Configure:0000/Run:0000/Filtered:0000/std::string/no_src/time']
        self.assertTrue(len(filteredMsgsEventIds)==2 and
                        filteredMsgsEventIds['seconds'][0]==1364147551 and
                         filteredMsgsEventIds['seconds'][1]==1364147551 and
                         filteredMsgsEventIds['nanoseconds'][0]==107587445 and
                         filteredMsgsEventIds['nanoseconds'][1]==174323092, 
                        msg="time dataset not right in Filtered:0000/std::string/no_src/time")

        # we don't write damage to the filtered groups:
        with self.assertRaises(KeyError):
            f['/Configure:0000/Run:0000/Filtered:0000/std::string/no_src/_damage']
        with self.assertRaises(KeyError):
            f['/Configure:0000/Run:0000/Filtered:0000/std::string/no_src/_mask']

        if self.cleanUp:
            os.unlink(output_h5)
        
    def test_ndarrays_allWrittenToFile(self):
        '''check that all ndarrays and strings written to the file
        '''
        input_file = TESTDATA_T1
        output_h5 = os.path.join(OUTDIR,'unit_test_ndarrays_allWrittenToFile.h5')
        cfgfile = writeCfgFile(input_file,output_h5,"Translator.TestModuleNDArrayString Translator.H5Output")
        cfgfile.file.flush()
        self.runPsanaOnCfg(cfgfile,output_h5)
        f=h5py.File(output_h5,'r')

        double3D = f['/Configure:0000/Run:0000/CalibCycle:0000/NDArray/my_double3D/data'][...]
        float2Da = f['/Configure:0000/Run:0000/CalibCycle:0000/NDArray/my_float2Da/data'][...]
        float2Db = f['/Configure:0000/Run:0000/CalibCycle:0000/NDArray/my_float2Db/data'][...]
        int1D = f['/Configure:0000/Run:0000/CalibCycle:0000/NDArray/my_int1D/data'][...]
        str1=f['/Configure:0000/Run:0000/CalibCycle:0000/std::string/my_string1/data'][...]
        str2=f['/Configure:0000/Run:0000/CalibCycle:0000/std::string/my_string2/data'][...]

        output="double3D:\n%r\nfloat2Da:\n%r\nfloat2Db:\n%r\nint1D:\n%r\nstr1:\n%r\nstr2:\n%r\n" % (double3D, float2Da, float2Db, int1D, str1, str2)
        self.assertEqual(output,NDARRAY_2EVENTS, "output of all nd arrays wrong,\n *** expected: ***\n%s\n*** produced: ***\n%s\n" % (NDARRAY_2EVENTS, output))
        if self.cleanUp:
            os.unlink(output_h5)

    def test_filter_all_ndarray(self):
        '''check that we can filter out all ndarrays and strings
        '''
        input_file = TESTDATA_T1
        output_h5 = os.path.join(OUTDIR, "unit_test_filter_all_ndarray.h5")
        cfgfile = writeCfgFile(input_file,output_h5,"Translator.TestModuleNDArrayString Translator.H5Output")
        # first add any options to H5Output, then other modules
        cfgfile.write("ndarray_types = exclude\n")
        cfgfile.write("std_string = exclude\n")
        cfgfile.file.flush()
        self.runPsanaOnCfg(cfgfile,output_h5)
        f=h5py.File(output_h5,'r')

        # none of the ndarrays or strings should be in here:
        with self.assertRaises(KeyError):
            f['/Configure:0000/Run:0000/CalibCycle:0000/NDArray/my_double3D']
        with self.assertRaises(KeyError):
            f['/Configure:0000/Run:0000/CalibCycle:0000/NDArray/my_float2Da']
        with self.assertRaises(KeyError):
            f['/Configure:0000/Run:0000/CalibCycle:0000/NDArray/my_float2Db']
        with self.assertRaises(KeyError):
            f['/Configure:0000/Run:0000/CalibCycle:0000/NDArray/my_int1D']
        with self.assertRaises(KeyError):
            f['/Configure:0000/Run:0000/CalibCycle:0000/std::string/my_string1']
        with self.assertRaises(KeyError):
            f['/Configure:0000/Run:0000/CalibCycle:0000/std::string/my_string2']

        if self.cleanUp:
            os.unlink(output_h5)

    def test_filter_some_ndarray_exclude(self):
        '''check that we can filter some ndarrays by excluding them
        '''
        input_file = TESTDATA_T1
        output_h5 = os.path.join(OUTDIR, "unit_test_filter_some_ndarray_exclude.h5")
        cfgfile = writeCfgFile(input_file,output_h5,"Translator.TestModuleNDArrayString Translator.H5Output")
        # first add any options to H5Output, then other modules
        cfgfile.write("ndarray_key_filter = exclude my_int1D my_float2Da\n")
        cfgfile.write("std_string_key_filter = exclude my_string1\n")
        cfgfile.file.flush()
        self.runPsanaOnCfg(cfgfile,output_h5,extraOpts='',printPsanaOutput=False)
        f=h5py.File(output_h5,'r')

        # now these should not be here
        with self.assertRaises(KeyError):
            f['/Configure:0000/Run:0000/CalibCycle:0000/NDArray/my_int1D']
        with self.assertRaises(KeyError):
            f['/Configure:0000/Run:0000/CalibCycle:0000/NDArray/my_float2Da']
        with self.assertRaises(KeyError):
            f['/Configure:0000/Run:0000/CalibCycle:0000/std::string/my_string1']
        # but these should
        self.assertEqual(len(f['/Configure:0000/Run:0000/CalibCycle:0000/NDArray/my_double3D/data']),2)
        self.assertEqual(len(f['/Configure:0000/Run:0000/CalibCycle:0000/NDArray/my_float2Db/data']),2)
        self.assertEqual(len(f['/Configure:0000/Run:0000/CalibCycle:0000/std::string/my_string2/data']),2)

        if self.cleanUp:
            os.unlink(output_h5)

    def test_filter_some_ndarray_include(self):
        '''check that we can filter some ndarrays by including them
        '''
        input_file = TESTDATA_T1
        output_h5 = os.path.join(OUTDIR, "unit_test_filter_some_ndarray_include.h5")
        cfgfile = writeCfgFile(input_file,output_h5,"Translator.TestModuleNDArrayString Translator.H5Output")
        # first add any options to H5Output, then other modules
        cfgfile.write("ndarray_key_filter = include my_int1D my_float2Da\n")
        cfgfile.write("std_string_key_filter = include my_string1\n")
        cfgfile.file.flush()
        self.runPsanaOnCfg(cfgfile,output_h5)
        f=h5py.File(output_h5,'r')

        # now these should not be here
        with self.assertRaises(KeyError):
            f['/Configure:0000/Run:0000/CalibCycle:0000/NDArray/my_double3D']
        with self.assertRaises(KeyError):
            f['/Configure:0000/Run:0000/CalibCycle:0000/NDArray/my_float2Db']
        with self.assertRaises(KeyError):
            f['/Configure:0000/Run:0000/CalibCycle:0000/std::string/my_string2']

        # but these should
        self.assertEqual(len(f['/Configure:0000/Run:0000/CalibCycle:0000/NDArray/my_int1D/data']),2)
        self.assertEqual(len(f['/Configure:0000/Run:0000/CalibCycle:0000/NDArray/my_float2Da/data']),2)
        self.assertEqual(len(f['/Configure:0000/Run:0000/CalibCycle:0000/std::string/my_string1/data']),2)

        if self.cleanUp:
            os.unlink(output_h5)

    def test_src_filter_include(self):
        '''check src filtering
        '''
        grpSrcList = [('EvrData::DataV3', 'NoDetector.0:Evr.0'),
                        ('Bld::BldDataEBeamV3','EBeam'),
                        ('Bld::BldDataPhaseCavity','PhaseCavity'),
                        ('Bld::BldDataFEEGasDetEnergy','FEEGasDetEnergy'),
                        ('Ipimb::DataV2/XppSb2_Ipm','XppSb2_Ipm'),
                        ('Ipimb::DataV2/XppSb3_Ipm','XppSb3_Ipm')]
        for group,src in grpSrcList:
            output_h5 = os.path.join(OUTDIR,"unit-test_src_filter_include.h5")
            cfgfile = writeCfgFile(TESTDATA_T1,output_h5)
            cfgfile.write("src_filter = include %s\n" % src)
            cfgfile.file.flush()
            self.runPsanaOnCfg(cfgfile,output_h5)
            f=h5py.File(output_h5,'r')
            
            fullGroup = '/Configure:0000/Run:0000/CalibCycle:0000/%s' % group
            h5group = f[fullGroup]  # should not throw exception

            filteredGroups = [grpSrc[0] for grpSrc in grpSrcList if grpSrc[1] != src]
            for filteredGroup in filteredGroups:
                with self.assertRaises(KeyError):
                    f['/Configure:0000/Run:0000/CalibCycle:0000/%s' % filteredGroup]

        if self.cleanUp:
            os.unlink(output_h5)
        
    def test_src_filter_exclude(self):
        '''check src filtering
        '''
        grpSrcList = [('EvrData::DataV3', 'NoDetector.0:Evr.0'),
                        ('Bld::BldDataEBeamV3','EBeam'),
                        ('Bld::BldDataPhaseCavity','PhaseCavity'),
                        ('Bld::BldDataFEEGasDetEnergy','FEEGasDetEnergy'),
                        ('Ipimb::DataV2/XppSb2_Ipm','XppSb2_Ipm'),
                        ('Ipimb::DataV2/XppSb3_Ipm','XppSb3_Ipm')]
        for group,src in grpSrcList:
            output_h5 = os.path.join(OUTDIR,"unit-test_src_filter_exclude.h5")
            cfgfile = writeCfgFile(TESTDATA_T1,output_h5)
            cfgfile.write("src_filter = exclude %s\n" % src)
            cfgfile.file.flush()
            self.runPsanaOnCfg(cfgfile,output_h5)
            f=h5py.File(output_h5,'r')
            
            fullGroup = '/Configure:0000/Run:0000/CalibCycle:0000/%s' % group
            with self.assertRaises(KeyError):
                h5group = f[fullGroup] 

            includedGroups = [grpSrc[0] for grpSrc in grpSrcList if grpSrc[1] != src]
            for includedGroup in includedGroups:
                grp = f['/Configure:0000/Run:0000/CalibCycle:0000/%s' % includedGroup]

        if self.cleanUp:
            os.unlink(output_h5)

    def test_newWriter(self):
        '''check newWriter capability
        '''
        output_h5 = os.path.join(OUTDIR,"unit-test_newwriter.h5")
        cfgfile = writeCfgFile(TESTDATA_T1,output_h5,
                               moduleList='Translator.TestNewHdfWriter Translator.H5Output')
        cfgfile.file.flush()
        self.runPsanaOnCfg(cfgfile,output_h5)
        f=h5py.File(output_h5,'r')
            
        MyData = f['/Configure:0000/Run:0000/CalibCycle:0000/Translator::MyData/Translator.TestNewHdfWriter/data']
        self.assertEqual(len(MyData),2)
        self.assertEqual(MyData['eventCounter'][0],1)
        self.assertEqual(MyData['eventCounter'][1],2)

        if self.cleanUp:
            os.unlink(output_h5)

    def test_type_filter(self):
        '''check that the type_filter switch works
        '''
        output_file = os.path.join(OUTDIR,"unit-test-type_filter.h5")
        cfgfile = writeCfgFile(TESTDATA_T1, output_file)
        cfgfile.write("store_epics = no\n")
        cfgfile.write("type_filter = exclude psana\n")
        cfgfile.file.flush()
        self.runPsanaOnCfg(cfgfile,output_file)
        h5 = h5py.File(output_file,'r')
        self.assertEqual(h5.keys(),['Configure:0000'])
        self.assertEqual(h5['Configure:0000'].keys(),['Run:0000'])
        self.assertEqual(h5['Configure:0000/Run:0000'].keys(),['CalibCycle:0000'])
        self.assertEqual(h5['Configure:0000/Run:0000/CalibCycle:0000'].keys(),[])

        # now check that if we exclude psana we see ndarrays
        cfgfile = writeCfgFile(TESTDATA_T1, output_file,
                               moduleList='Translator.TestModuleNDArrayString Translator.H5Output')
        cfgfile.write("type_filter = exclude psana\n")
        cfgfile.write("overwrite = true\n")
        cfgfile.file.flush()
        self.runPsanaOnCfg(cfgfile,output_file)
        h5 = h5py.File(output_file,'r')
        nddata = h5['/Configure:0000/Run:0000/CalibCycle:0000/NDArray/my_double3D/data']
        self.assertEqual(len(nddata),2)

        # now check that type_filter will pick out a few types:
        cfgfile = writeCfgFile(TESTDATA_T1, output_file)
        cfgfile.write("type_filter = include IpmFex Evr\n")
        cfgfile.write("store_epics = no\n")
        cfgfile.write("overwrite = true\n")
        cfgfile.file.flush()
        self.runPsanaOnCfg(cfgfile,output_file)
        cmd = 'h5ls -r %s | grep data' % output_file
        p = sb.Popen(cmd, shell=True, stdout=sb.PIPE, stderr=sb.PIPE)
        o,e = p.communicate()
        self.assertEqual(e,'')
        hasTypes = [ ln.find('EvrData')>=0 or ln.find('IpmFex')>=0 for ln in o.strip().split('\n') ]
        self.assertTrue(all(hasTypes), "all lines are EvrData or IpmFex")
        if self.cleanUp:
            os.unlink(output_file)
        
    def test_partition(self):
        '''check that partition type is getting written
        test_051 has partition object
        '''
        input_file = TESTDATA_PARTITION
        output_h5 = os.path.join(OUTDIR,"unit-test_partition.h5")
        cfgfile = writeCfgFile(input_file, output_h5)
        self.runPsanaOnCfg(cfgfile,output_h5, extraOpts='-n 1',printPsanaOutput=self.printPsanaOutput)
        cfgfile.close()
        f = h5py.File(output_h5,'r')
        cfg=f['Configure:0000/Partition::ConfigV1/Control/config']
        value=cfg.value
        expected=(67108871L, 8L)
        self.assertEqual(value[0],expected[0],msg="bldMask wrong: read=%s expected=%s" % (value[0],expected[0]) )
        self.assertEqual(value[1],expected[1],msg="number of groups: read=%s expected=%s" % (value[1],expected[1]) )
        sources=f['Configure:0000/Partition::ConfigV1/Control/sources']
        expectedSources = [((16788226L, 256L), 0L),
                           ((16796544L, 184551937L), 1L),
                           ((100682534L, 0L), 0L),
                           ((100682534L, 1L), 0L),
                           ((100682534L, 2L), 0L),
                           ((100682534L, 26L), 0L),
                           ((16796457L, 184552706L), 2L),
                           ((16796017L, 184552707L), 3L)]

        for source,expected in zip(sources,expectedSources):
            self.assertEqual(source[0][0],expected[0][0], "source logical wrong")
            self.assertEqual(source[0][1],expected[0][1], "source physical wrong")
            self.assertEqual(source[1],expected[1], "group wrong")
        if self.cleanUp:
            os.unlink(output_h5)
        
    def test_alias(self):
        '''check that aliases are getting created.

        test_050 has an aliasCfg object in the config, the aliases are:
        numAlias=7 
        0: EXS_OPAL -> SxrBeamline.0:Opal1000.1 
        1: Laser_OPAL_CVD02 -> SxrEndstation.0:Opal1000.2 
        2: Scienta_OPAL_CVD01A -> SxrEndstation.0:Opal1000.0 
        3: TSS_OPAL -> SxrBeamline.0:Opal1000.0 
        4: XES_OPAL_CVD01B -> SxrEndstation.0:Opal1000.1 
        5: acq01 -> SxrEndstation.0:Acqiris.0 
        6: acq02 -> SxrEndstation.0:Acqiris.2

        During the first 120 events that are in the test_050 file, we get data from 
        SxrEndstation.0:Acqiris.0 and SxrEndstation.0:Acqiris.2 but not the Opal1000
        sources. So one can test for those links.
        '''
        input_file = TESTDATA_ALIAS
        output_h5 = os.path.join(OUTDIR,"unit-test_alias.h5")
        cfgfile = writeCfgFile(input_file, output_h5)
        self.runPsanaOnCfg(cfgfile,output_h5, extraOpts='-n 10',printPsanaOutput=self.printPsanaOutput)
        cfgfile.close()

        # check for acq01 soft link:
        cmd = 'h5ls -r %s | grep acq01' % output_h5
        p = sb.Popen(cmd,shell=True,stdout=sb.PIPE,stderr=sb.PIPE)
        o,e = p.communicate()
        assert e == ""
        o=o.strip()
        expectedOutput = '''/Configure:0000/Acqiris::ConfigV1/acq01 Soft Link {SxrEndstation.0:Acqiris.0}
/Configure:0000/Run:0000/CalibCycle:0000/Acqiris::DataDescV1/acq01 Soft Link {SxrEndstation.0:Acqiris.0}'''
        self.assertEqual(o,expectedOutput, msg="acq01 alias output for cmd=%s mismatch, expected=\n'%s'\nobserved=\n'%s'" % (cmd,expectedOutput,o))

        # check for acq02 soft link:
        cmd = 'h5ls -r %s | grep acq02' % output_h5
        p = sb.Popen(cmd,shell=True,stdout=sb.PIPE,stderr=sb.PIPE)
        o,e = p.communicate()
        assert e == ""
        o=o.strip()
        expectedOutput = '''/Configure:0000/Acqiris::ConfigV1/acq02 Soft Link {SxrEndstation.0:Acqiris.2}
/Configure:0000/Run:0000/CalibCycle:0000/Acqiris::DataDescV1/acq02 Soft Link {SxrEndstation.0:Acqiris.2}'''
        self.assertEqual(o,expectedOutput, msg="acq02 alias output for cmd=%s mismatch, expected=\n'%s'\nobserved=\n'%s'" % (cmd,expectedOutput,o))

        # check that the Alias::ConfigV1 object got written:
        h5file=h5py.File(output_h5,'r')
        aliases=h5file['/Configure:0000/Alias::ConfigV1/Control/aliases']
        srcLog = aliases['src']['log']
        srcPhy = aliases['src']['phy']
        aliasName = aliases['aliasName']
        srcLogExpected=np.array([16782900, 16783349, 16779118, 16783350, 16779116, 16790217, 16790229], dtype=np.uint32)
        srcPhyExpected=np.array([184550145, 201327362, 201327360, 184550144, 201327361, 201327104, 201327106], dtype=np.uint32)
        aliasNameExpected=np.array(['EXS_OPAL', 'Laser_OPAL_CVD02', 'Scienta_OPAL_CVD01A', 'TSS_OPAL', 'XES_OPAL_CVD01B', 'acq01', 'acq02'], dtype='|S31')
        self.assertTrue(all(aliasName == aliasNameExpected), msg="Alias::ConfigV1 aliasName mismatch, expected=\n'%s'\nobserved=\n'%s'" % (aliasNameExpected,aliasName))
        self.assertTrue(all(srcPhy == srcPhyExpected), msg="Alias::ConfigV1 srcPhy mismatch, expected=\n'%s'\nobserved=\n'%s'" % (srcLog,srcLogExpected))
        self.assertTrue(all(srcLog == srcLogExpected), msg="Alias::ConfigV1 srcLog mismatch, expected=\n'%s'\nobserved=\n'%s'" % (srcPhy,srcPhyExpected))
        
        # check that if we turn off aliases, we do not see them:
        cfgfile = writeCfgFile(input_file, output_h5)
        cfgfile.write("create_alias_links=false\n")
        cfgfile.flush()
        self.runPsanaOnCfg(cfgfile,output_h5, extraOpts='-n 10', printPsanaOutput=self.printPsanaOutput)
        cfgfile.close()
        cmd = 'h5ls -r %s | grep "acq02\|acq01"' % output_h5
        p = sb.Popen(cmd,shell=True,stdout=sb.PIPE,stderr=sb.PIPE)
        o,e = p.communicate()
        self.assertEqual(e.strip(),"",msg="Error running h5ls on %s, err=%s"  % (output_h5,e))
        self.assertEqual(o.strip(),"",msg="acq01 or acq02 still in file: %s, h5ls output=%s" % (output_h5,o))
        if self.cleanUp:
            os.unlink(output_h5)

        # check that if we uses aliases in the src_filter option, that it works
        cfgfile = writeCfgFile(input_file, output_h5)
        cfgfile.write("src_filter=exclude acq01 acq02\n")
        cfgfile.flush()
        self.runPsanaOnCfg(cfgfile,output_h5, extraOpts='-n 10', printPsanaOutput=self.printPsanaOutput)
        cfgfile.close()
        cmd = 'h5ls -r %s | grep "SxrEndstation.0:Acqiris.0\|SxrEndstation.0:Acqiris.2"' % output_h5
        p = sb.Popen(cmd,shell=True,stdout=sb.PIPE,stderr=sb.PIPE)
        o,e = p.communicate()
        self.assertEqual(e.strip(),"",msg="There was a problem running h5ls: stderr=%s"%e.strip())
        self.assertEqual(o.strip(),"",msg="The src's were not filtered by the aliases, output of h5ls -r is: %s" % o)
        if self.cleanUp:
            os.unlink(output_h5)
        
                
#  run unit tests when imported as a main module
#
#  To run an individual test, do something like (from the release directory)
#  python -m unittest Translator.testing.H5Output.test_file_t1_translated_correctly

if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], '-v'])
