#--------------------------------------------------------------------------
# Description:
#   Test script for psana_test
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
from psana_test import epicsPvToStr
import psana_test.psanaTestLib as ptl
import psana

DATADIR = ptl.getTestDataDir()
OUTDIR = "data/psana_test"

#------------------
# Utility functions 
#------------------
def getLinesBeforeAndAfterPos(string,pos,linesBefore,linesAfter):
    '''returns the positions in the string for the newline 
    characters that are linesBefore earlier in string from pos, and
    linesAfter past pos in string 
    '''
    linesBefore = min(1,linesBefore)
    linesAfter = min(1,linesAfter)
    startPos = pos
    for idx in range(linesBefore):
        startPos = string.rfind('\n',0,startPos)
        if startPos == -1:
            startPos = 0
            break
    endPos = pos
    for idx in range(linesAfter):
        endPos = string.find('\n',endPos)
        if endPos == -1:
            endPos = len(string)
            break
    return startPos, endPos

def getH5OutfileName(path):
    basename = os.path.basename(path)
    h5basename = os.path.splitext(basename)[0] + '.h5'
    return os.path.join(OUTDIR,h5basename)

#-------------------------------
#  Unit test class definition --
#-------------------------------
class Psana( unittest.TestCase ) :

    def setUp(self) :
    	""" 
    	Method called to prepare the test fixture. This is called immediately 
    	before calling the test method; any exception raised by this method 
    	will be considered an error rather than a test failure.  
    	"""
        assert os.path.exists(DATADIR), "Data dir: %s does not exist, cannot run unit tests" % DATADIR
        assert os.path.exists(OUTDIR), "Output directory: %s does not exist, can't run unit tests" % OUTDIR
        self.cleanUp = True    # delete intermediate files if True
        self.verbose = False    # print psana output, ect

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

    def runPsanaOnCfg(self,cfgfile=None,cmdLineOptions='', errorCheck=True, linesBefore=10, linesAfter=5):
        '''Runs psana, takes cfgfile object as well as cmdLineOptions.

        If errorCheck is True it tests that lower case psana output does not include: fatal, error,
                        segmentation fault, seg falut, traceback

        returns the pair of stdout, stderr from the psana run
        '''
        assert cfgfile is not None or cmdLineOptions is not '', "one of cfgfile or cmdLineOptions must be set"
        assert isinstance(cmdLineOptions,str), "extraOpts for psana command line is %r, not a str" % cmdLineOptions
        cfgFileStr = ''
        if cfgfile is not None:
            cfgfile.flush()
            cfgFileStr = '-c %s' % cfgfile.name        
        psana_cmd = "psana %s %s" % (cmdLineOptions,cfgFileStr)
        p = sb.Popen(psana_cmd,shell=True,stdout=sb.PIPE, stderr=sb.PIPE)
        o,e = p.communicate()
        if self.verbose:
            print "===== psana cmd ======"
            print psana_cmd
            if cfgfile is not None:
                print "===== psana cfg file ===="
                print cfgfile.read()
            print "===== psana stdout ===="
            print o
            print "===== psana stderr ===="
            print e
            sys.stdout.flush()
        if errorCheck:
            for output,source in zip([o,e],['stdout','stderr']):
                lowerOutput = output.lower()
                for errorTerm in ['fatal', 'error', 'segmentation fault', 'seg fault', 'traceback']:
                    pos = lowerOutput.find(errorTerm)
                    if pos >= 0:
                        startPos, endPos = getLinesBeforeAndAfterPos(lowerOutput,pos,linesBefore,linesAfter)
                        self.assertTrue(False,msg="'%s' found in psana output: ...\n%s\n..." % \
                                        (errorTerm, output[startPos:endPos]))
        return o,e

    def h5Translate(self, inFile, outFile, cmdLineOptions=''):
        cmdLine = cmdLineOptions
        cmdLine += " -m Translator.H5Output"
        cmdLine += " -o Translator.H5Output.output_file=%s" % outFile
        cmdLine += " -o Translator.H5Output.overwrite=True"
        cmdLine += " %s" % inFile
        self.assertTrue(os.path.exists(inFile), "test data file: %s not found" % inFile)
        try:
            fin = file(inFile,'r')
        except:
            self.assertTrue(False,msg="test data exists, but this program cannot read it")
        fin.close()        
        self.assertTrue(os.path.exists(os.path.split(outFile)[0]),msg="output directory does not exist")
        try:
            fout = file(outFile,'w')
        except:
            self.assertTrue(False,msg="program cannot write the file: %s check for permission issues" % outFile)
        fout.close()
        os.unlink(outFile)
        self.runPsanaOnCfg(cmdLineOptions=cmdLine)
        self.assertTrue(os.path.exists(outFile), msg="Translation did not produce outfile: %s" % outFile)
        
    def test_MoreRecentEpicsStored(self):
        '''When the same epics pv is recorded from several sources, or several times in the same source, 
        the most recent one should be stored. test_073 is a case where this occurs, and before the code
        was changed to add the most recent one, it was the earlier one that was stored.
        The earlier one, from pvid 192, has stamp.sec=767233751 stamp.nsec= 40108031 
        while the later one, from pvid 9    stamp.sec=767233751 stamp.nsec=140115967
        '''
        TEST_73 = os.path.join(DATADIR,'test_073_cxi_cxid5514_e423-r0049-s00-c00.xtc')
        assert os.path.exists(TEST_73), "input file: %s does not exist, can't run test" % TEST_73
        psana.setConfigFile('')
        ds = psana.DataSource(TEST_73)
        epicsStore = ds.env().epicsStore()
        ds.events().next()  # advance to event 0
        pvName = 'CXI:R56:SHV:VHS2:CH1:CurrentMeasure'
        pv = epicsStore.getPV(pvName)
        self.assertFalse(pv is None, msg="could not get %s from epics store" % pvName)
        self.assertEqual(pv.stamp().nsec(), 140115967, msg="pv %s does not have expected nano-seconds"  % pvName)
        self.assertEqual(pv.stamp().sec(), 767233751, msg="pv %s does not have expected seconds"  % pvName)

    def test_EpicsIssues1(self):
        '''Test a number of issues:
        * That a epics pv that is accidentally masked by an alias is accessible. 
          In the test file, test_010, there is an alias CXI:SC2:MZM:09:ENCPOSITIONGET that masks that pv. 
          Check that the pv is accessible.
        * For hdf5 input, pv's with the same pvid from different sources are read back 
          properly.
        * For hdf5 input, pv's with the same pvid from different sources cannot both have aliases
          (this is a current limitation) test that only one alias is available. If this limitiation is
          fixed, this test should be changed to test both aliases are available and work properly.
        * The aliases available for the xtc input and the hdf5 input will be different. Test for
          expected values (aliases in the xtc should be available when psana processes the xtc
          except for aliases that have the same name as an existing pv).
          In hdf5, additional aliases get removed as per the above test, can't have two aliases with
          the same pvId (current limitation, change test if fixed).
        '''
        TEST_10 = os.path.join(DATADIR,'test_010_cxi_cxia4113_e325-r0002-s00-c00.xtc')

        def checkAliases(self, aliases, estore, label):
            for source, aliasDict in aliases.iteritems():
                for alias, pvnamePvIdPair in aliasDict.iteritems():
                    pvname,pvId = pvnamePvIdPair
                    aliasPv = estore.getPV(alias)
                    pv = estore.getPV(pvname)
                    if aliasPv is not None:
                        self.assertFalse(pv is None, 
                                         msg="%s: source=%s\n can only get pv: %s through alias: %s" % (label, source, pvname, alias))
                        aliasStr = epicsPvToStr(aliasPv)
                        pvStr = epicsPvToStr(pv)
                        self.assertEqual(aliasStr,pvStr,
                                         msg="%s: source=%s\n alias and pv data disagree for alias=%s pv=%s\n%s\%s" % (label, source, alias, pv, aliasStr, pvStr))
                        if hasattr(pv, 'pvName'):
                            self.assertEqual(pvname, pv.pvName(),msg="%s: source=%s\n pvName=%s disagrees with pvName=%s in epics data" % (label, source, pvname,pv.pvName()))

        # some aliases in the xtc file for TEST_010
        aliases={}
        #                                      alias                                     pvName             pvId
        aliases['EpicsArch.0:NoDevice.1']={'KB1 Horiz Foucssing Mirror Roll and Pitch':('CXI:KB1:MMS:07.RBV',0),
                                           'KB1 Horiz Foucssing Mirror Roll and Pitch-CXI:KB1:MMS:08.RBV':('CXI:KB1:MMS:08.RBV',1),
                                           'KB1 Vert Foucssing Mirror Pitch':('CXI:KB1:MMS:11.RBV',2),
                                           'KB1 Vert Foucssing Mirror Pitch-CXI:SC2:MZM:08:ENCPOSITIONGET':('CXI:SC2:MZM:08:ENCPOSITIONGET',3),
                                           'KB1 Vert Foucssing Mirror Pitch-CXI:SC2:MZM:09:ENCPOSITIONGET':('CXI:SC2:MZM:09:ENCPOSITIONGET',4)
                                           }
        aliases['EpicsArch.0:NoDevice.0']={'e-beam duration in fs':('SIOC:SYS0:ML01:AO971',  0),
                                           'x-ray beam duration in fs':('SIOC:SYS0:ML01:AO972',  1),
                                           'x-ray power in GW, where the power is defined as x-ray pulse en':('SIOC:SYS0:ML01:AO973',  2),
                                           'DG3 Spectrometer':('CXI:USR:MMS:20.RBV',  3),
                                           'DG3 Spectrometer-CXI:DG3:PIC:01.RBV':('CXI:DG3:PIC:01.RBV',  4),
                                           'DG3 Spectrometer-CXI:DG3:PIC:02.RBV':('CXI:DG3:PIC:02.RBV',  5),
                                           'DG3 Spectrometer-CXI:DG3:PIC:03.RBV':('CXI:DG3:PIC:03.RBV',  6),
                                           'DG3 Spectrometer-CXI:SC2:PIC:04.RBV':('CXI:SC2:PIC:04.RBV',  7),
                                           'DG3 Spectrometer-CXI:SC2:PIC:05.RBV':('CXI:SC2:PIC:05.RBV',  8),
                                           'DG3 Spectrometer-CXI:SC2:PIC:06.RBV':('CXI:SC2:PIC:06.RBV',  9),
                                           'fine motors + focus':('CXI:USR:MMS:09.RBV', 10),
                                           'fine motors + focus-CXI:USR:MMS:10.RBV':('CXI:USR:MMS:10.RBV', 11),
                                           'fine motors + focus-CXI:USR:MMS:11.RBV':('CXI:USR:MMS:11.RBV', 12),
                                           'fine motors + focus-CXI:USR:MMS:12.RBV':('CXI:USR:MMS:12.RBV', 13),
                                           'LEDs':('CXI:USR:SC2:ANLINOUT:00:DEVICE:NAME', 14),
                                           'CXI:SC2:MZM:07:ENCPOSITIONGET':('CXI:SC2:MZM:07:ENCPOSITIONGET', 44),
                                           'CXI:SC2:MZM:08:ENCPOSITIONGET':('CXI:SC2:MZM:08:ENCPOSITIONGET', 45),
#                                           'CXI:SC2:MZM:09:ENCPOSITIONGET':('CXI:SC2:MZM:10:ENCPOSITIONGET', 46),
                                           'CXI:SC2:MZM:09:ENCPOSITIONGET-CXI:SC2:MZM:12:ENCPOSITIONGET':('CXI:SC2:MZM:12:ENCPOSITIONGET', 47),
                                           '140k Y Stepper':('CXI:DG2:MMS:17.RBV',225),
                                           'DG4 IPM/PIM':('CXI:DG4:MMS:01.RBV',226),
                                           'DG4 IPM/PIM-CXI:DG4:MMS:02.RBV':('CXI:DG4:MMS:02.RBV',227),
                                           'DG4 IPM/PIM-CXI:DG4:MMS:03.RBV':('CXI:DG4:MMS:03.RBV',228),
                                           'DG4 IPM/PIM-CXI:DG4:MMS:04.RBV':('CXI:DG4:MMS:04.RBV',229),
                                           'DG4 IPM/PIM-CXI:DG4:MMS:05.RBV':('CXI:DG4:MMS:05.RBV',230),
                                       }
        h5_outfile = getH5OutfileName(TEST_10)
        self.h5Translate(TEST_10, h5_outfile, cmdLineOptions='-n 1')

        # this is the pv that is masked by an alias
        src1_pvid_4 = {'pvname':'CXI:SC2:MZM:09:ENCPOSITIONGET', 
                       'beginJobValue':0.0,
                       'event0value':0.0,
                       'event0stamp':(743137317, 444140000),
                       'alias':'KB1 Vert Foucssing Mirror Pitch-CXI:SC2:MZM:09:ENCPOSITIONGET'}
        # this is a different pv from a different source that has the same pvid
        src0_pvid_4 = {'pvname':'CXI:DG3:PIC:01.RBV', 
                       'beginJobValue':1.0005e-02,
                       'event0value':1.0005e-02,
                       'event0stamp':(742173122, 724943000),
                       'alias': 'DG3 Spectrometer-CXI:DG3:PIC:01.RBV'}
        # test xtc
        psana.setConfigFile('')
        dsXtc = psana.DataSource(TEST_10)
        estore = dsXtc.env().epicsStore()
        # the alias 'CXI:SC2:MZM:09:ENCPOSITIONGET' to the pv 'CXI:SC2:MZM:10:ENCPOSITIONGET'
        # masks the original pv, psana should be removing this alias:
        self.assertEqual('',estore.alias('CXI:SC2:MZM:10:ENCPOSITIONGET'), msg="xtc: psana is not removing alias 'CXI:SC2:MZM:09:ENCPOSITIONGET' for 'CXI:SC2:MZM:10:ENCPOSITIONGET' that masks pv with alias name")
        # are pv's there
        pv0 = estore.getPV(src0_pvid_4['pvname'])
        pv1 = estore.getPV(src1_pvid_4['pvname'])
        self.assertTrue(pv0 is not None, msg="xtc: src0 pvid4 pvname=%s not found during beginJob" % src0_pvid_4['pvname'])
        self.assertTrue(pv1 is not None, msg="xtc: src1 pvid4 pvname=%s not found during beginJob" % src1_pvid_4['pvname'])
        # right value in beginJob?
        self.assertEqual(pv0.value(0), src0_pvid_4['beginJobValue'],msg="xtc: src0 pvid4 pvname=%s beginJob value wrong" % src0_pvid_4['pvname'])
        self.assertEqual(pv1.value(0), src1_pvid_4['beginJobValue'],msg="xtc: src1 pvid4 pvname=%s beginJob value wrong" % src1_pvid_4['pvname'])
        # are aliases there?
        alias0 = estore.getPV(src0_pvid_4['alias'])
        alias1 = estore.getPV(src1_pvid_4['alias'])
        self.assertTrue(alias0 is not None, msg="xtc: src0 pvid4 alias=%s not found during beginJob" % src0_pvid_4['alias'])
        self.assertTrue(alias1 is not None, msg="xtc: src1 pvid4 alias=%s not found during beginJob" % src1_pvid_4['alias'])
        # do aliases have right value?
        self.assertEqual(alias0.value(0), src0_pvid_4['beginJobValue'],msg="xtc: src0 pvid4 pvname=%s beginJob value wrong from alias=%s" % (src0_pvid_4['pvname'], src0_pvid_4['alias']))
        self.assertEqual(alias1.value(0), src1_pvid_4['beginJobValue'],msg="xtc: src1 pvid4 pvname=%s beginJob value wrong from alias=%s" % (src1_pvid_4['pvname'], src1_pvid_4['alias']))
        # check expected number of aliases and pvNames, have not verified that they are all correct, 199 and 227
        # are what was observed when test was written
        self.assertEqual(len(estore.aliases()), 199, msg="xtc: estore does not have expected number of aliases")
        self.assertEqual(len(estore.pvNames()), 227, msg="xtc: estore does not have expected number of pvNames")
        checkAliases(self, aliases, estore, "xtc configure")
        # go to the next event
        dsXtc.events().next()
        pv0 = estore.getPV(src0_pvid_4['pvname'])
        pv1 = estore.getPV(src1_pvid_4['pvname'])
        self.assertTrue(pv0 is not None, msg="xtc: src0 pvid4 pvname=%s not found during event 0" % src0_pvid_4['pvname'])
        self.assertTrue(pv1 is not None, msg="xtc: src1 pvid4 pvname=%s not found during event 0" % src1_pvid_4['pvname'])
        self.assertEqual(pv0.value(0), src0_pvid_4['event0value'],msg="xtc: src0 pvid4 pvname=%s event0 value wrong" % src0_pvid_4['pvname'])
        self.assertEqual(pv1.value(0), src1_pvid_4['event0value'],msg="xtc: src1 pvid4 pvname=%s event0 value wrong" % src1_pvid_4['pvname'])
        self.assertEqual(pv0.stamp().sec(), src0_pvid_4['event0stamp'][0],msg="xtc: src0 pvid4 pvname=%s event0 stamp sec wrong" % src0_pvid_4['pvname'])
        self.assertEqual(pv1.stamp().sec(), src1_pvid_4['event0stamp'][0],msg="xtc: src1 pvid4 pvname=%s event0 stamp sec wrong" % src1_pvid_4['pvname'])
        self.assertEqual(pv0.stamp().nsec(), src0_pvid_4['event0stamp'][1],msg="xtc: src0 pvid4 pvname=%s event0 stamp sec wrong" % src0_pvid_4['pvname'])
        self.assertEqual(pv1.stamp().nsec(), src1_pvid_4['event0stamp'][1],msg="xtc: src1 pvid4 pvname=%s event0 stamp sec wrong" % src1_pvid_4['pvname'])
        alias0 = estore.getPV(src0_pvid_4['alias'])
        alias1 = estore.getPV(src1_pvid_4['alias'])
        checkAliases(self, aliases, estore, "configure")
        self.assertTrue(alias0 is not None, msg="xtc: src0 pvid4 alias=%s not found during event0" % src0_pvid_4['alias'])
        self.assertTrue(alias1 is not None, msg="xtc: src1 pvid4 alias=%s not found during event0" % src1_pvid_4['alias'])
        self.assertEqual(alias0.value(0), src0_pvid_4['event0value'],msg="xtc: src0 pvid4 pvname=%s beginJob value wrong from alias=%s" % (src0_pvid_4['pvname'], src0_pvid_4['alias']))
        self.assertEqual(alias1.value(0), src1_pvid_4['event0value'],msg="xtc: src1 pvid4 pvname=%s beginJob value wrong from alias=%s" % (src1_pvid_4['pvname'], src1_pvid_4['alias']))
        del alias1
        del alias0
        del pv1
        del pv0
        del estore
        del dsXtc
        
        psana.setConfigFile('')
        dsH5 = psana.DataSource(h5_outfile) 
        estore = dsH5.env().epicsStore()
        # check expected number of aliases and pvNames, have not verified that they are all correct, 193 and 227
        # are what was observed when test was written
        self.assertEqual(len(estore.aliases()), 193, msg="h5: estore does not have expected number of aliases")
        self.assertEqual(len(estore.pvNames()), 227, msg="h5: estore does not have expected number of pvNames")
        checkAliases(self, aliases, estore, "h5 configure")
        # are pv's there with right value?
        pv0 = estore.getPV(src0_pvid_4['pvname'])
        pv1 = estore.getPV(src1_pvid_4['pvname'])
        self.assertTrue(pv0 is not None, msg="h5: src0 pvid4 pvname=%s not found during beginJob" % src0_pvid_4['pvname'])
        self.assertTrue(pv1 is not None, msg="h5: src1 pvid4 pvname=%s not found during beginJob" % src1_pvid_4['pvname'])
        self.assertEqual(pv0.value(0), src0_pvid_4['beginJobValue'],msg="h5: src0 pvid4 pvname=%s beginJob value wrong" % src0_pvid_4['pvname'])
        self.assertEqual(pv1.value(0), src1_pvid_4['beginJobValue'],msg="h5: src1 pvid4 pvname=%s beginJob value wrong" % src1_pvid_4['pvname'])
        alias0 = estore.getPV(src0_pvid_4['alias'])
        alias1 = estore.getPV(src1_pvid_4['alias'])
        # these aliases share the same pvid, due to limitation in psana-translate, there should only be one of them
        self.assertTrue((alias0 is None) or (alias1 is None), msg="h5: one of the src0 pvid4 and scr1 pvid4 aliases is not none")
        self.assertFalse((alias0 is None) and (alias1 is None), msg="h5: both the scr0 pvid4 and src1 pvid4 aliases are none")
        # go to the next event
        dsH5.events().next()
        pv0 = estore.getPV(src0_pvid_4['pvname'])
        pv1 = estore.getPV(src1_pvid_4['pvname'])
        self.assertTrue(pv0 is not None, msg="src0 pvid4 pvname=%s not found during event 0" % src0_pvid_4['pvname'])
        self.assertTrue(pv1 is not None, msg="src1 pvid4 pvname=%s not found during event 0" % src1_pvid_4['pvname'])
        self.assertEqual(pv0.value(0), src0_pvid_4['event0value'],msg="src0 pvid4 pvname=%s event0 value wrong" % src0_pvid_4['pvname'])
        self.assertEqual(pv1.value(0), src1_pvid_4['event0value'],msg="src1 pvid4 pvname=%s event0 value wrong" % src1_pvid_4['pvname'])
        self.assertEqual(pv0.stamp().sec(), src0_pvid_4['event0stamp'][0],msg="src0 pvid4 pvname=%s event0 stamp sec wrong" % src0_pvid_4['pvname'])
        self.assertEqual(pv1.stamp().sec(), src1_pvid_4['event0stamp'][0],msg="src1 pvid4 pvname=%s event0 stamp sec wrong" % src1_pvid_4['pvname'])
        self.assertEqual(pv0.stamp().nsec(), src0_pvid_4['event0stamp'][1],msg="src0 pvid4 pvname=%s event0 stamp sec wrong" % src0_pvid_4['pvname'])
        self.assertEqual(pv1.stamp().nsec(), src1_pvid_4['event0stamp'][1],msg="src1 pvid4 pvname=%s event0 stamp sec wrong" % src1_pvid_4['pvname'])

        if self.cleanUp: os.unlink(h5_outfile)


    def test_s80merge(self):
        '''tests if the s80 stream is merged properly        
        '''
        DAQ_fiducials_with_matching_s80 = [0x0DAE5,   # s0
                                           0x0DAFD,   # s0
                                           0x0DADF,   # s1
                                           0x0DAD9,   # s5
                                           0x0DAEB,   # s5
                                           0x0DAF7    # s5
                                           ]
        DAQ_fiducials_with_no_matching_s80 = [0x0D95C,  # s0
                                              0x0DAD6,  # s0
                                              0x13137,  # s1
                                              0x0D956, # s1
                                              0x0DAD3, # s1
                                              0x0DAF4, # s1
                                              0x1312B, # s1
                                              0x0D95F, # s5
                                              0x0DAC1, # s5
                                              0x1313A, # s5
                                              0x0E805, # s5
                                              0x0E80E, # s0
                                             ]
        s80_fiducials_with_no_DAQ = [56049, 56067, 78243]

        def eventHasDaqAndS80Data(evt, opal0, opal1, opal2, orca, xtcav):
            if evt.get(psana.Camera.FrameV1, opal0) is None: return False
            if evt.get(psana.Camera.FrameV1, opal1) is None: return False
            if evt.get(psana.Camera.FrameV1, opal2) is None: return False
            if evt.get(psana.Camera.FrameV1, orca)  is None: return False
            if evt.get(psana.Camera.FrameV1, xtcav) is None: return False
            return True
            
        def eventHasOnlyDaq(evt, opal0, opal1, opal2, orca, xtcav):
            if evt.get(psana.Camera.FrameV1, opal0) is None: return False
            if evt.get(psana.Camera.FrameV1, opal1) is None: return False
            if evt.get(psana.Camera.FrameV1, opal2) is None: return False
            if evt.get(psana.Camera.FrameV1, orca) is None: return False
            if evt.get(psana.Camera.FrameV1, xtcav) is not None: return False
            return True
            
        def eventHasOnlyS80(evt, opal0, opal1, opal2, orca, xtcav):
            if evt.get(psana.Camera.FrameV1, opal0) is not None: return False
            if evt.get(psana.Camera.FrameV1, opal1) is not None: return False
            if evt.get(psana.Camera.FrameV1, opal2) is not None: return False
            if evt.get(psana.Camera.FrameV1, orca)  is not None: return False
            if evt.get(psana.Camera.FrameV1, xtcav) is None: return False
            return True

        psana.setConfigFile('')
        dataSourceDir = os.path.join(ptl.getMultiFileDataDir(),'test_004_xppa1714')
        ds = psana.DataSource('exp=xppa1714:run=157:dir=%s' % dataSourceDir)
        # by using the aliases, we test that psana is processing the alias list from
        # both the s80 and the DAQ streams
#        opal0 = psana.Source('DetInfo(XppEndstation.0:Opal1000.0)')
#        opal1 = psana.Source('DetInfo(XppEndstation.0:Opal1000.1)')
#        opal2 = psana.Source('DetInfo(XppEndstation.0:Opal1000.2)')
#        orca  = psana.Source('DetInfo(XppEndstation.0:OrcaFl40.0)')
#        xtcav = psana.Source('DetInfo(XrayTransportDiagnostic.0:Opal1000.0)')
        opal0 = psana.Source('opal_0')
        opal1 = psana.Source('opal_1')
        opal2 = psana.Source('opal_2')
        orca  = psana.Source('orca')
        xtcav = psana.Source('xtcav')
        for calibNumber, calibIter in enumerate(ds.steps()):
            for eventNumber, evt in enumerate(calibIter.events()):
                eventId = evt.get(psana.EventId)
                fid = eventId.fiducials()
                if fid in DAQ_fiducials_with_matching_s80:
                    self.assertTrue(eventHasDaqAndS80Data(evt, opal0, opal1, opal2, orca, xtcav),
                                    msg="fid=%s should have both DAQ and s80" % fid)
                elif fid in DAQ_fiducials_with_no_matching_s80:
                    self.assertTrue(eventHasOnlyDaq(evt, opal0, opal1, opal2, orca, xtcav),
                                    "fid=%s should have only DAQ data" % fid)
                elif fid in s80_fiducials_with_no_DAQ:
                    self.assertTrue(eventHasOnlyS80(evt, opal0, opal1, opal2, orca, xtcav),
                                    "fid=%s should have only s80" % fid)
            self.assertTrue((calibNumber,eventNumber) in [(0,16), (1,8)], 
                            msg="should be 16 events in calib 0, and 8 in calib 1")

    def test_mp(self):
        '''parallel child process mode
        
        For testing, I'll just run one process with psana_test.dump to get all the
        output in one file. For some reason, I frequently get the error message

        Standard exception caught in runApp(): ExceptionErrno: writing to ready pipe failed: Broken pipe [in function next at PSXtcMPInput/src/DgramSourceWorker.cpp:84

        when I run psana_test.dump in parallel mode. Something flakely about the piping.
        If often seems that I don't get the error message if I run with debug output.
        
        So to test, I'm running with debug output, and saving the dump into a separate
        file that I compare.
        '''

        dataSourceDir = os.path.join(ptl.getMultiFileDataDir(), 'test_004_xppa1714')

       # test that mp mode gives us what we saw before on DAQ only streams
        dumpOutput = 'unittest_test_mp_mpmode.dump'
        cmd = '''psana -c '' -p 1'''
        cmd += ' -o psana_test.dump.output_file=%s' % dumpOutput
        cmd += (''' -m psana_test.dump exp=xppa1714:run=157:stream=0-20:dir=%s''' % dataSourceDir)
        o,e = ptl.cmdTimeOut(cmd,100)
        dumpOutput += '.subproc_0'
        md5 = ptl.get_md5sum(dumpOutput)
        prev_md5 = 'fbd1b3a999adb4cdef882c7aceb356dd'
        failMsg  = 'prev md5=%s\n' % prev_md5
        failMsg += 'curr md5=%s\n' % md5
        failMsg += 'are not equal. cmd:\n'
        failMsg += cmd
        self.assertEqual(prev_md5, md5, msg=failMsg)
        os.unlink(dumpOutput)
        
        # test that mp mode is the same as not mp mode (DAQ only streams)
        dumpOutput = 'unittest_test_mp_normal.dump'
        cmd = '''psana -c '' -o psana_test.dump.output_file=%s''' % dumpOutput
        cmd += (''' -m psana_test.dump exp=xppa1714:run=157:stream=0-20:dir=%s''' % dataSourceDir)
        o,e = ptl.cmdTimeOut(cmd,100)
        md5 = ptl.get_md5sum(dumpOutput)
        failMsg  = 'prev md5=%s\n' % prev_md5
        failMsg += 'curr md5=%s\n' % md5
        failMsg += 'are not equal. cmd:\n'
        failMsg += cmd
        self.assertEqual(prev_md5, md5, msg=failMsg)
        os.unlink(dumpOutput)
        
        
if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], '-v'])
