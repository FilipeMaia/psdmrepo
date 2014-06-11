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
import psana

DATADIR = "/reg/g/psdm/data_test/Translator"
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

if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], '-v'])
