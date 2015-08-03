#--------------------------------------------------------------------------
# Description:
#   Test script for ParCorAna
#   
#------------------------------------------------------------------------


#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os
import logging
import tempfile
import unittest
from cStringIO import StringIO
import numpy as np
import h5py
import glob
import shutil
#-----------------------------
# Imports for other modules --
#-----------------------------
import psana
from AppUtils.AppDataPath import AppDataPath
import psana_test.psanaTestLib as ptl

import ParCorAna as corAna

### helper function
def runCmd(cmd, verbose=True):
    o,e,retcode = ptl.cmdTimeOutWithReturnCode(cmd)
    if verbose: print "---  ran cmd: %s" % cmd
    return retcode

def removeAllInProgressFromParentDir(fname):
    basedir, basename = os.path.split(fname)
    assert len(basedir)>0 and os.path.exists(basedir)
    inProgressFiles = glob.glob(os.path.join(basedir, "*.inprogress"))
    for inProgress in inProgressFiles:
        os.unlink(inProgress)

def unindent(x):
    def numSpacesStart(ln):
        n=0
        while len(ln)>0 and ln[0]==' ':
            ln = ln[1:]
            n+= 1
        return n
    lns = x.split('\n')
    allLeadingSpaces = [numSpacesStart(ln) for ln in lns if len(ln.strip())>0]
    minLeadingSpaces = min(allLeadingSpaces)
    return '\n'.join([ln[minLeadingSpaces:] for ln in lns])

class FormatFileName( unittest.TestCase ) :

    def setUp(self) :
        self.longMessage = True
        destDirBase = AppDataPath(os.path.join("ParCorAna","testingDir")).path()
        self.tempDestDir = tempfile.mkdtemp(dir=destDirBase)

    def tearDown(self) :
        shutil.rmtree(self.tempDestDir, ignore_errors=True)
        
    def test_formatFileName(self):
        fname = os.path.join(self.tempDestDir, "file.h5")
        fname_w_T = os.path.join(self.tempDestDir, "file_%T.h5")
        fname_w_C = os.path.join(self.tempDestDir, "file_%C.h5")
        fname_other = os.path.join(self.tempDestDir, "file_jnk.h5")

        self.assertEqual(corAna.formatFileName(fname),fname)

        tmfname = corAna.formatFileName(fname_w_T)
        os.system('touch %s' % tmfname)
        self.assertNotEqual(tmfname,fname)        
                                              # %C 2015 05 05 16 19 59
        self.assertEqual(len(tmfname),len(fname_w_T)-2   +4 +2 +2 +2 +2 +2, msg="tmfname=%s" % tmfname)

        os.system('touch %s' % fname)
        os.system('touch %s' % tmfname)

        c0 = corAna.formatFileName(fname_w_C)

        self.assertNotEqual(c0,fname)
        self.assertEqual(c0, fname_w_C.replace('%C','000'))
        os.system('touch %s' % c0)

        c1 = corAna.formatFileName(fname_w_C)
        self.assertEqual(c1, fname_w_C.replace('%C','001'))

        os.system('touch %s' % c1)
        os.system('touch %s' % fname_other)

        c2 = corAna.formatFileName(fname_w_C)
        self.assertEqual(c2, fname_w_C.replace('%C','002'))

class ParCorAna( unittest.TestCase ) :

    def setUp(self) :
        self.longMessage = True

    def tearDown(self) :
        pass
        
    def test_parseDataSetString(self):
        '''test parseDataSetString function
        '''
        dsOpts = corAna.parseDataSetString('exp=amo123:run=12')
        self.assertEqual(dsOpts['exp'],'amo123')
        self.assertEqual(dsOpts['run'],[12])
        self.assertEqual(dsOpts['h5'],False)
        self.assertEqual(dsOpts['xtc'],True)
        self.assertEqual(dsOpts['live'],False)
        self.assertEqual(dsOpts['shmem'],False)
        
    def test_noConfig(self):
        system_params = {}
        user_params = {}
        test_alt = False
        self.assertRaises(AssertionError, corAna.CommSystemFramework, system_params, user_params, test_alt)

    def test_logger(self):
        msg1 = 'hi there'
        msg2 = 'what?'
        try:
            stdout = sys.stdout
            stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            self.assertRaises(AssertionError, corAna.makeLogger, False, True, True, True, 0, 'INFO', False)
            l = corAna.makeLogger( False, True, False, False, 0, 'INFO', False)
            l2 = corAna.makeLogger( False, True, False, False, 0, 'INFO', False) # make sure getting another ref doesn't double handlers
            l.info(msg1)
            l.warning(msg2)
        except Exception,e:
            sys.stdout = stdout
            sys.stderr = stderr
            raise e

        stderrLns = [ln for ln in sys.stderr.getvalue().split('\n') if len(ln.strip())>0]
        stdoutLns = [ln for ln in sys.stdout.getvalue().split('\n') if len(ln.strip())>0]
        sys.stderr.close()
        sys.stdout.close()
        sys.stdout = stdout
        sys.stderr = stderr

        self.assertEqual(len(stderrLns),2)
        self.assertEqual(len(stdoutLns),0)

        self.assertTrue(stderrLns[0].find('INFO')>0 and stderrLns[0].find(msg1)>0, msg='log ln=%s does not have INFO nor %s in it' % (stderrLns[0], msg1))
        self.assertTrue(stderrLns[1].find('WARNING')>0 and stderrLns[1].find(msg2)>0, msg='log ln=%s does not have WARNING nor %s in it' % (stderrLns[1], msg2))


class Cspad2x2( unittest.TestCase ) :
    '''Test on small cspad2x2.
    This test data has 60 events. 
    The order we go through the data depends on whether or not we are 
    round robining through the servers, or picking the earliest server. 
    One fiducial is missing from the events. This is fiducial 0x08214. 
    Assigning a 0 up 120hz counter to the events, this will be counter 56. 
    You will see events 0,1,2, ...,55,57,58,60.
    When you go through the events using three servers for the three streams in round robin
    you get the events in this order:
    0 5 3 1 7 4 2 8 9 6 11 12 10 17 15 13 18 19 14 21 24 16 22 25 20 26 29 23 30 31 27 34 32 28 36 37 33 39 42 35 41 43 38 44 48 40 47 49 45 50 54 46 52 55 51 57 58 53 60 59
    
    For incremental window below, with a window of 20, at the end of the run we will 
    be looking at these counters to form delays:
    41 43 38 44 48 40 47 49 45 50 54 46 52 55 51 57 58 53 60 59

    below we sort them, and give the differences between them:
    38 40 41 43 44 45 46 47 48 49 50 51 52 53 54 55 57 58 59 60
      2  1  2  1  1  1  1  1  1  1  1  1  1  1  1  2  1  1  1 

    for these delays: [1,2,3,5,7,10,15,23,34,50] we will expect counts of
    1:16, 2:
    '''
    def setUp(self) :
        dataDir = os.path.join(ptl.getMultiFileDataDir(), 'test_013_xcsi0314')
        experiment = 'xcsi0314'
        run = 178  
        correctVersion = 0

        maskColorDir = os.path.join(dataDir, 'maskColorDir')
        correctOutputDir = os.path.join(dataDir, 'ParCorAnaTestAnswers')
        assert os.path.exists(maskColorDir)
        assert os.path.exists(correctOutputDir)
        maskFileBaseName = '%s-r%d_XcsEndstation_0_Cspad2x2_0_mask_ndarrCoords.npy' % (experiment, run)
        testMaskFileBaseName = '%s-r%d_XcsEndstation_0_Cspad2x2_0_testmask_ndarrCoords.npy' % (experiment, run)
        colorFileBaseName =  '%s-r%d_XcsEndstation_0_Cspad2x2_0_color_ndarrCoords.npy' % (experiment, run)
        finecolorFileBaseName =  '%s-r%d_XcsEndstation_0_Cspad2x2_0_finecolor_ndarrCoords.npy' % (experiment, run)
        atEndCorrectBaseName = 'g2calc_cspad2x2_atEnd_%s-r%4.4d_v%d.h5' % (experiment, run, correctVersion)
        accumCorrectBaseName = 'g2calc_cspad2x2_incrAccum_%s-r%4.4d_v%d.h5' % (experiment, run, correctVersion)
        windowNoRoundRobinCorrectBaseName = 'g2calc_cspad2x2_windowNoRoundRobin_%s-r%4.4d_v%d.h5' % (experiment, run, correctVersion)
        
        maskFile = os.path.join(maskColorDir, maskFileBaseName)
        testMaskFile = os.path.join(maskColorDir, testMaskFileBaseName)
        colorFile = os.path.join(maskColorDir, colorFileBaseName)
        finecolorFile = os.path.join(maskColorDir, finecolorFileBaseName)
        self.atEndAnswerFile = os.path.join(correctOutputDir, atEndCorrectBaseName)
        self.accumAnswerFile = os.path.join(correctOutputDir, accumCorrectBaseName)
        self.windowNoRoundRobinAnswerFile = os.path.join(correctOutputDir, windowNoRoundRobinCorrectBaseName)
        assert os.path.exists(maskFile), "mask file %s doesn't exist" % maskFile
        assert os.path.exists(testMaskFile),  "test maskfile %s doesn't exist" % testMaskFile
        assert os.path.exists(colorFile),  "color file %s doesn't exist" % colorFile
        assert os.path.exists(finecolorFile),  "fine color file %s doesn't exist" % finecolorFile
        assert os.path.exists(self.atEndAnswerFile), "atEnd file %s doesn't exist" % self.atEndFile
        assert os.path.exists(self.accumAnswerFile), "accumAnswerFile file %s doesn't exist" % self.accumAnswerFile
        assert os.path.exists(self.windowNoRoundRobinAnswerFile), "window no round robin file %s doesn't exist" % self.windowNoRoundRobinAnswerFile

        numServers = 3
        serversRoundRobin = False
        
        # make a random directory for the testing that we will remove when done
        destDirBase = AppDataPath(os.path.join("ParCorAna","testingDir")).path()
        assert len(destDirBase)>0, "did not find testingDir base dir in the ParCorAna data dir"
        tempDestDir = tempfile.mkdtemp(dir=destDirBase)
        if not os.path.exists(tempDestDir): os.mkdir(tempDestDir)
        h5outputBaseName = 'g2calc_cspad2x2_%%s_%s-r%4.4d.h5' % (experiment, run)  # has %%s for for testName
        testH5outputBaseName = 'test_' + h5outputBaseName
        h5outputFile = os.path.join(tempDestDir, h5outputBaseName)
        testH5outputFile = os.path.join(tempDestDir, testH5outputBaseName)
        removeAllInProgressFromParentDir(h5outputFile)
        userClass = '--TESTS-MUST-FILL-THIS-IN--'
        testName = '--TESTS-MUST-FILL-THIS-IN--'
        numTimes = 100  # test data only has 60 events
        delays = [1, 2, 3, 5, 7, 10, 15, 23, 34, 50]
        saturatedValue = (1<<15) 
        update = 0
        calibDir = ptl.getTestCalibDir()
        self.formatDict = locals().copy()

        self.numEvents = 60     # see docstring in CsPad2x2 class about
        # number of events and counter from 0,1,2 ...,55, 57,58, ..., 60
        # That means we expect the following delay counts:
        def expectedDelay(delay):
            if delay > 4: return 60 - delay
            return 59-delay
        self.expectedCounts = [expectedDelay(delay) for delay in delays]

        self.tempDestDir = tempDestDir
        self.dataDir = dataDir
        self.run = run

        self.configFileContent='''
        import psana
        import numpy as np
        import ParCorAna as corAna

        system_params = {{}}
        system_params['dataset']   = 'exp={experiment}:run={run}:dir={dataDir}'
        system_params['src']       = 'DetInfo(XcsEndstation.0:Cspad2x2.0)'
        system_params['psanaType'] = psana.CsPad2x2.ElementV1 
        system_params['ndarrayProducerOutKey'] = 'ndarray'
        system_params['ndarrayCalibOutKey'] = 'calibrated'
        system_params['psanaOptions'], system_params['outputArrayType'] = \\
                corAna.makePsanaOptions(srcString=system_params['src'],
                                   psanaType=system_params['psanaType'],
                                   ndarrayOutKey=system_params['ndarrayProducerOutKey'],
                                   ndarrayCalibOutKey=system_params['ndarrayCalibOutKey'])

        system_params['psanaOptions']['psana.calib-dir']='{calibDir}'
        system_params['workerStoreDtype']=np.float32
        system_params['maskNdarrayCoords'] = '{maskFile}'
        system_params['testMaskNdarrayCoords'] = '{testMaskFile}'
        system_params['numServers'] = {numServers}
        system_params['serversRoundRobin'] = {serversRoundRobin}
        system_params['serverHosts'] = None  # None means system selects which hosts to use (default). 
        system_params['times'] = {numTimes}
        system_params['update'] = {update}
        system_params['delays'] = {delays}
        testName = '{testName}'
        system_params['h5output'] = '{h5outputFile}' % testName
        system_params['testH5output'] = '{testH5outputFile}' % testName
        system_params['overwrite'] = True
        system_params['verbosity'] = 'INFO'
        system_params['numEvents'] = 0
        system_params['testNumEvents'] = 0


        import ParCorAna.UserG2 as UserG2
        system_params['userClass'] = {userClass}

        user_params = {{}}
        user_params['colorNdarrayCoords'] = '{colorFile}'
        user_params['colorFineNdarrayCoords'] = '{finecolorFile}'
        user_params['saturatedValue'] = {saturatedValue}
        user_params['plot_colors'] = None
        user_params['print_delay_curves'] = False
        user_params['LLD'] = 1E-9
        user_params['notzero'] = 1E-5
        user_params['psmon_plot'] = False
        user_params['ipimb_threshold_lower'] = .05
        user_params['ipimb_srcs'] = []
        user_params['debug_plot']=False
        user_params['plot_colors']=[1,4,6,8]
        user_params['iX']=None
        user_params['iY']=None

        '''

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
        shutil.rmtree(self.tempDestDir, ignore_errors=True) 


    def test_FilesSame(self):
        '''
        check that the input files haven't changed
        '''
        md5sums={'maskColorDir/xcsi0314-r178_XcsEndstation_0_Cspad2x2_0_color_ndarrCoords.npy':     'dad6ebe25b364eeea4114c036b54ea4c',
                 'maskColorDir/xcsi0314-r178_XcsEndstation_0_Cspad2x2_0_finecolor_ndarrCoords.npy': 'f6cdb19b26d28d96a17b87ddde3be12c',
                 'maskColorDir/xcsi0314-r178_XcsEndstation_0_Cspad2x2_0_mask_ndarrCoords.npy':      '9b8ade01f93fc087228c15cad9944856', 
                 'maskColorDir/xcsi0314-r178_XcsEndstation_0_Cspad2x2_0_testmask_ndarrCoords.npy':  '282715e77fb5e4247a6b0851f3b244ea', 
                 'e524-r0178-s00-c00.xtc':                                                          'b73a43ee4393c8c793d430f951cad021', 
                 'e524-r0178-s01-c00.xtc':                                                          'eee2248370bef1a94202d5d6afd89799', 
                 'e524-r0178-s02-c00.xtc':                                                          'd340d899c5ab36f34b75df419af3b711', 
                 'e524-r0178-s03-c00.xtc':                                                          '111d1ab55c6bbb685bea7d5501587e1d', 
                 'e524-r0178-s04-c00.xtc':                                                          '18fcbc6eec20d2a94f31750f49dc1bda', 
                 'e524-r0178-s05-c00.xtc':                                                          '9d87909f0c613ca6433fc94d0985521d',
                 'ParCorAnaTestAnswers/g2calc_cspad2x2_atEnd_xcsi0314-r0178_v0.h5':                 '777d665671ce0b38476c16377f597724',
                 'ParCorAnaTestAnswers/g2calc_cspad2x2_incrAccum_xcsi0314-r0178_v0.h5':             'f3e67511e46fcff5aab272463faeccfc',

                 # these two files have serversRoundRobin = True, but the unit tests are now testing with
                 # round robin = False, except for the one windowed test
#                 'ParCorAnaTestAnswers/g2calc_cspad2x2_atEnd_xcsi0314-r0178_v1.h5':                 'c32335ba2c43a4b91e3f0f3477ff02db',
#                 'ParCorAnaTestAnswers/g2calc_cspad2x2_incrAccum_xcsi0314-r0178_v1.h5':             '48d129663088942661ca3f6ca6dcc8a5',

                 'ParCorAnaTestAnswers/g2calc_cspad2x2_windowNoRoundRobin_xcsi0314-r0178_v0.h5':    '112d83b2d0e7ee545e26b07a27005fa9',
        }
        for fname, prev_md5 in md5sums.iteritems():
            fullFname = os.path.join(self.dataDir,fname)
            assert os.path.exists(fullFname)
            cur_md5 = ptl.get_md5sum(fullFname)
            self.assertEqual(cur_md5, prev_md5, msg="md5 has changed for %s. old=%s new=%s" % \
                             (fullFname, prev_md5, cur_md5))
        
    def writeConfigFile(self, configname):
        configFileName = os.path.join(self.tempDestDir, configname)
        configFile = file(configFileName, 'w')
        configFile.write(unindent(self.configFileContent.format(**self.formatDict)))
        configFile.close()
        return configFileName

    def checkDelays(self, h5fileName, delays, expectedCounts):
        h5file = h5py.File(h5fileName,'r')
        systemDelays = list(h5file['system/system_params/delays'][:])
        userDelays = list(h5file['user/G2_results_at_000060/delays'][:])
        self.assertListEqual(delays, systemDelays, msg='delays written to config != system delays')
        self.assertListEqual(systemDelays, userDelays, msg="in h5 output file, system and user section do not have same delays")
        counts = list(h5file['user/G2_results_at_000060/delay_counts'][:])
        self.assertEqual(len(counts), len(expectedCounts))
        self.assertListEqual(counts, expectedCounts, msg="delay counts wrong.\nAns=%r\nRes=%r\nDly=%r" % \
                             (expectedCounts,  counts, list(delays)))

    def test_G2atEnd(self):
        self.formatDict['userClass']='UserG2.G2atEnd'
        testName = 'atEnd'
        self.formatDict['testName'] = testName
        configFileName = self.writeConfigFile('config_G2atEnd.py')

        cmd = 'mpiexec -n 9 parCorAnaDriver --test_main -c ' + configFileName
        self.assertEqual(0, runCmd(cmd, verbose=True), msg="Error running %s" % cmd)

        # check delays
        h5outputFile = self.formatDict['h5outputFile'] % testName
        self.checkDelays(h5outputFile , self.formatDict['delays'], self.expectedCounts)

        # check that the output agrees with the previously saved version:
        cmdCmpPrevious = 'cmpParCorAnaH5OutputPy -i serversRoundRobin %s %s' % (h5outputFile, self.atEndAnswerFile)
        self.assertEqual(0, runCmd(cmdCmpPrevious, verbose=True), msg="Error checking against previously saved output, cmd %s" % cmdCmpPrevious)

        cmd = 'parCorAnaDriver --test_alt -c ' + configFileName
        self.assertEqual(0, runCmd(cmd, verbose=True), msg="Error running %s" % cmd)

        cmd = 'parCorAnaDriver --cmp -c ' + configFileName
        self.assertEqual(0, runCmd(cmd, verbose=True), msg="error running %s - files must differ" % cmd)
        
    def test_SaturatedPixels(self):
        self.formatDict['userClass']='UserG2.G2atEnd'
        self.formatDict['saturatedValue'] = 300
        self.formatDict['update'] = 30
        testName = 'atEndSat'
        self.formatDict['testName'] = testName
        configFileName = self.writeConfigFile('config_G2atEndSat.py')

        cmd = 'mpiexec -n 9 parCorAnaDriver --test_main -c ' + configFileName
        output,err,retcode = ptl.cmdTimeOutWithReturnCode(cmd)
        self.assertEqual(0, retcode, msg="Error running %s" % cmd)
        output += '\n'
        output += err
        expectedLines = ["99 new saturated pixels being removed from color labeling. 1 colors are being dropped",
                         "1 new saturated pixels being removed from color labeling. 1 colors are being dropped"]

        for ln in output.split('\n'):
            if len(expectedLines)==0:
                break
            if ln.find(expectedLines[0])>=0:
                expectedLines.pop(0)
                
        self.assertEqual(0,len(expectedLines), "From cmd: %s\nDid not find output lines that included these lines in this order:\n%s" % \
                         (cmd, '\n'.join(expectedLines)))

        
    def test_G2IncrementalAccumulator(self):
        self.formatDict['userClass']='UserG2.G2IncrementalAccumulator'
        testName = 'incrAccum'
        self.formatDict['testName'] = testName
        configFileName = self.writeConfigFile('config_G2IncrementalAccumulator.py')

        cmd = 'mpiexec -n 9 parCorAnaDriver --test_main -c ' + configFileName
        self.assertEqual(0, runCmd(cmd, verbose=True), msg="Error running %s" % cmd)

        # check delays
        h5outputFile = self.formatDict['h5outputFile'] % testName
        self.checkDelays(h5outputFile, self.formatDict['delays'], self.expectedCounts)

        # check that the output agrees with the previously saved version:
        cmdCmpPrevious = 'cmpParCorAnaH5OutputPy -i serversRoundRobin %s %s' % (h5outputFile, self.accumAnswerFile)
        self.assertEqual(0, runCmd(cmdCmpPrevious, verbose=True), msg="Error checking against previously saved output, cmd %s" % cmdCmpPrevious)

        cmd = 'parCorAnaDriver --test_alt -c ' + configFileName
        self.assertEqual(0, runCmd(cmd, verbose=True), msg="Error running %s" % cmd)

        cmd = 'parCorAnaDriver --cmp -c ' + configFileName
        self.assertEqual(0, runCmd(cmd, verbose=True), msg="error running %s - files must differ" % cmd)
        
    def test_G2WindowRoundRobin(self):
        self.formatDict['userClass']='UserG2.G2IncrementalWindowed'
        testName = 'windowRoundRobin'
        self.formatDict['testName'] = testName
        self.formatDict['numTimes'] = 20 
        self.formatDict['serversRoundRobin'] = True
        self.assertListEqual(self.formatDict['delays'],[1, 2, 3, 5, 7,10,15,23,34,50])
        self.expectedCounts =               [ 18, 17, 16, 15, 13, 10, 5, 0, 0, 0]
        configFileNameA = self.writeConfigFile('config_G2windowRoundRobin.py')
        cmd = 'mpiexec -n 9 parCorAnaDriver --test_main -c ' + configFileNameA
        self.assertEqual(0, runCmd(cmd, verbose=True), msg="Error running %s" % cmd)
        # check delays
        h5outputFile = self.formatDict['h5outputFile'] % testName
        self.checkDelays(h5outputFile, self.formatDict['delays'], self.expectedCounts)
        # compare the round robin result to non-round robin results
        nonRoundRobin = self.windowNoRoundRobinAnswerFile
        cmd = 'cmpParCorAnaH5OutputPy -i serversRoundRobin %s %s' % (h5outputFile, nonRoundRobin)
        o,e,retcode = ptl.cmdTimeOutWithReturnCode(cmd)
        self.assertEqual(0, retcode, msg="comparing windowRoundRobin with no round robin with numTimes=%d failed.\ncmp cmd=%s" % \
                         (self.formatDict['numTimes'], cmd))
        
    def test_G2WindowNoRoundRobin(self):
        self.formatDict['userClass']='UserG2.G2IncrementalWindowed'
        testName = 'windowNoRoundRobina'
        self.formatDict['testName'] = testName
        self.formatDict['numTimes'] = 20   # 60 events, so we will get a smaller window
        delays = self.formatDict['delays']
        self.assertListEqual(delays,[1,2,3,5,7,10,15,23,34,50])        
        # the last 20 long window (see comments in CsPad2x2 class) will be
        # 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 -- 57 58 59 60
        # with the missing 56, the expected counts will be
        self.expectedCounts =  [ 18, 17, 16, 15, 13, 10, 5, 0, 0, 0]
        configFileNameA = self.writeConfigFile('config_G2windowNoRoundRobin.py')

        cmd = 'mpiexec -n 9 parCorAnaDriver --test_main -c ' + configFileNameA
        self.assertEqual(0, runCmd(cmd, verbose=True), msg="Error running %s" % cmd)

        # check delays
        h5outputFile = self.formatDict['h5outputFile'] % testName
        self.checkDelays(h5outputFile, self.formatDict['delays'], self.expectedCounts)

        cmd = 'parCorAnaDriver --test_alt -c ' + configFileNameA
        self.assertEqual(0, runCmd(cmd, verbose=True), msg="Error running %s" % cmd)

        cmd = 'parCorAnaDriver --cmp -c ' + configFileNameA
        self.assertEqual(0, runCmd(cmd, verbose=True), msg="error running %s - files must differ" % cmd)
        
        # we expect windowed incremental to produce the same result as G2 at end with a small numTimes
        self.formatDict['userClass']='UserG2.G2atEnd'
        self.formatDict['testName'] = 'windowedNoRoundRobinAtEndForCmp'
        configFileNameB = self.writeConfigFile('config_G2windowNoRoundRobinCmdAtEnd.py')

        cmd = 'mpiexec -n 9 parCorAnaDriver --test_main -c ' + configFileNameB
        self.assertEqual(0, runCmd(cmd, verbose=True), msg="Error running %s" % cmd)

        h5A = h5outputFile
        h5B = self.formatDict['h5outputFile'] % self.formatDict['testName']
        cmd = 'cmpParCorAnaH5OutputPy -i serversRoundRobin,userClass %s %s' % (h5A, h5B)
        print "running cmd=%s" % cmd
        o,e,retcode = ptl.cmdTimeOutWithReturnCode(cmd)
        self.assertEqual(0, retcode, msg="comparing windowNoRoundRobin to atEnd with numTimes=%d failed.\ncmp cmd=%s\nconfigA=%s\nconfigB=%s" % \
                         (self.formatDict['numTimes'], cmd, configFileNameA, configFileNameB))

class UtilFunctions( unittest.TestCase ) :

    def setUp(self) :
        pass

    def tearDown(self) :
        pass

    def test_delay(self):
        delays = corAna.makeDelayList(start=1,
                                      stop=15000, 
                                      num=100,
                                      spacing='log',  # can also be 'lin'
                                      logbase=10.0)
        badDelays = [x for x in delays if x <1 or x > 15000]
        self.assertEqual(len(badDelays),0)
        self.assertEqual(len(delays),100)

    def test_replaceSubsetsWithAverage(self):        
        A = np.array(range(15))
        A.resize((3,5))
        labels = np.array([1]*5 + [2]*5 + [3]*5, np.int)
        labels.resize((3,5))
        avgA = corAna.replaceSubsetsWithAverage(A,labels)
        self.assertEqual(avgA.shape, A.shape)
        for idx in range(5):
            self.assertAlmostEqual(avgA[0,idx], 2.0)
        for idx in range(5):
            self.assertAlmostEqual(avgA[1,idx], 7.0)
        for idx in range(5):
            self.assertAlmostEqual(avgA[2,idx], 12.0)
        self.assertEqual(avgA.dtype, np.float32)
        A = A.astype(np.float64)
        avgA = corAna.replaceSubsetsWithAverage(A,labels)
        for idx in range(5):
            self.assertAlmostEqual(avgA[0,idx], 2.0)
        for idx in range(5):
            self.assertAlmostEqual(avgA[1,idx], 7.0)
        for idx in range(5):
            self.assertAlmostEqual(avgA[2,idx], 12.0)
        self.assertEqual(avgA.dtype, np.float64)

    def test_replaceSubsetsWithAverageOddShapes(self):        
        A = np.array(range(15))
        A.resize((3,5))
        labels = np.array([0]*2 + [2]*5 + [4]*5 + [6]*3, np.int)
        labels.resize((3,5))
        avgA = corAna.replaceSubsetsWithAverage(A,labels)
        self.assertAlmostEqual(avgA[0,0], 0.5)
        self.assertAlmostEqual(avgA[0,1], 0.5)

        self.assertAlmostEqual(avgA[0,2], 4.0)
        self.assertAlmostEqual(avgA[0,3], 4.0)
        self.assertAlmostEqual(avgA[0,4], 4.0)
        self.assertAlmostEqual(avgA[1,0], 4.0)
        self.assertAlmostEqual(avgA[1,1], 4.0)

        self.assertAlmostEqual(avgA[1,2], 9.0)
        self.assertAlmostEqual(avgA[1,3], 9.0)
        self.assertAlmostEqual(avgA[1,4], 9.0)
        self.assertAlmostEqual(avgA[2,0], 9.0)
        self.assertAlmostEqual(avgA[2,1], 9.0)

        self.assertAlmostEqual(avgA[2,2], 13.0)
        self.assertAlmostEqual(avgA[2,3], 13.0)
        self.assertAlmostEqual(avgA[2,4], 13.0)

    def test_replaceSubsetsWithAverageOddShapesAndWithCounts(self):        
        A = np.array(range(15))
        A.resize((3,5))
        labels = np.array([0]*2 + [2]*5 + [4]*5 + [6]*3, np.int)
        labels.resize((3,5))
        label2count = {0:2,2:5,4:5,6:3}
        avgA = corAna.replaceSubsetsWithAverage(A,labels, label2count)
        self.assertAlmostEqual(avgA[0,0], 0.5)
        self.assertAlmostEqual(avgA[0,1], 0.5)

        self.assertAlmostEqual(avgA[0,2], 4.0)
        self.assertAlmostEqual(avgA[0,3], 4.0)
        self.assertAlmostEqual(avgA[0,4], 4.0)
        self.assertAlmostEqual(avgA[1,0], 4.0)
        self.assertAlmostEqual(avgA[1,1], 4.0)

        self.assertAlmostEqual(avgA[1,2], 9.0)
        self.assertAlmostEqual(avgA[1,3], 9.0)
        self.assertAlmostEqual(avgA[1,4], 9.0)
        self.assertAlmostEqual(avgA[2,0], 9.0)
        self.assertAlmostEqual(avgA[2,1], 9.0)

        self.assertAlmostEqual(avgA[2,2], 13.0)
        self.assertAlmostEqual(avgA[2,3], 13.0)
        self.assertAlmostEqual(avgA[2,4], 13.0)

if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], '-v'])
