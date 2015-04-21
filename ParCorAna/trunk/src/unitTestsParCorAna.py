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
#import stat
import tempfile
import unittest
from cStringIO import StringIO
#import subprocess as sb
#import collections
#import math
import numpy as np
import h5py
#import glob
import shutil
#-----------------------------
# Imports for other modules --
#-----------------------------
import psana
from AppUtils.AppDataPath import AppDataPath
import psana_test.psanaTestLib as ptl

#import h5py
#import psana_test.psanaTestLib as ptl

import ParCorAna as corAna

### helper function
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

class ParCorAna( unittest.TestCase ) :

    def setUp(self) :
    	""" 
    	Method called to prepare the test fixture. This is called immediately 
    	before calling the test method; any exception raised by this method 
    	will be considered an error rather than a test failure.  
    	"""
        pass

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

    def setUp(self) :
    	""" 
    	Method called to prepare the test fixture. This is called immediately 
    	before calling the test method; any exception raised by this method 
    	will be considered an error rather than a test failure.  
    	"""
        dataDir = os.path.join(ptl.getMultiFileDataDir(), 'test_013_xcsi0314')
        experiment = 'xcsi0314'
        run = 178  

        maskColorDir = os.path.join(dataDir, 'maskColorDir')
        maskFileBaseName = '%s-r%d_XcsEndstation_0_Cspad2x2_0_mask_ndarrCoords.npy' % (experiment, run)
        testMaskFileBaseName = '%s-r%d_XcsEndstation_0_Cspad2x2_0_testmask_ndarrCoords.npy' % (experiment, run)
        colorFileBaseName =  '%s-r%d_XcsEndstation_0_Cspad2x2_0_color_ndarrCoords.npy' % (experiment, run)
        maskFile = os.path.join(maskColorDir, maskFileBaseName)
        testMaskFile = os.path.join(maskColorDir, testMaskFileBaseName)
        colorFile = os.path.join(maskColorDir, colorFileBaseName)

        assert os.path.exists(maskFile), "mask file %s doesn't exist" % maskFile
        assert os.path.exists(testMaskFile),  "test maskfile %s doesn't exist" % testMaskFile
        assert os.path.exists(colorFile),  "color file %s doesn't exist" % colorkFile

        numServers = 1

        numTimes = 100  # test data only has 60 events

        # make a random directory for the testing that we will remove when done
        destDirBase = AppDataPath(os.path.join("ParCorAna","testingDir")).path()
        assert len(destDirBase)>0, "did not find testingDir base dir in the ParCorAna data dir"
#        tempDestDir = tempfile.mkdtemp(dir=destDirBase)
        tempDestDir = os.path.join(destDirBase, "mytest")

        h5outputBaseName = 'g2calc_cspad2x2_%s-r%4.4d.h5' % (experiment, run)
        testH5outputBaseName = 'test_' + h5outputBaseName
        h5outputFile = os.path.join(tempDestDir, h5outputBaseName)
        testH5outputFile = os.path.join(tempDestDir, testH5outputBaseName)
        userClass = 'UserG2.G2atEnd'

        self.formatDict = locals().copy()

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

        system_params['workerStoreDtype']=np.float32
        system_params['maskNdarrayCoords'] = '{maskFile}'
        system_params['testMaskNdarrayCoords'] = '{testMaskFile}'
        system_params['numServers'] = {numServers}
        system_params['serverHosts'] = None  # None means system selects which hosts to use (default). 
        system_params['times'] = {numTimes}
        system_params['update'] = 0
        system_params['delays'] = corAna.makeDelayList(start=1,
                                                       stop=int(system_params['times']/2.0), 
                                                       num=10,
                                                       spacing='log',  # can also be 'lin'
                                                       logbase=np.e)

        system_params['h5output'] = '{h5outputFile}'
        system_params['testH5output'] = '{testH5outputFile}'
        system_params['overwrite'] = True
        system_params['verbosity'] = 'INFO'
        system_params['numEvents'] = 0
        system_params['testNumEvents'] = 0


        import ParCorAna.UserG2 as UserG2
        system_params['userClass'] = {userClass}

        user_params = {{}}
        user_params['colorNdarrayCoords'] = '{colorFile}'
        user_params['saturatedValue'] = (1<<15)
        user_params['LLD'] = 1E-9
        user_params['notzero'] = 1E-5
        user_params['psmon_plot'] = False
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
        pass
#        shutil.rmtree(self.tempDestDir, ignore_errors=True)  DVD REMOVE

    def test_basicRun(self):
        configFileName = os.path.join(self.tempDestDir, "config.py")
        configFile = file(configFileName, 'w')
        configFile.write(unindent(self.configFileContent.format(**self.formatDict)))
        configFile.close()
        cmd = 'mpiexec -n 4 parCorAnaDriver -c %s' % configFileName
        o,e = ptl.cmdTimeOut(cmd)
        print "ran cmd: %s" % cmd
        print "output=%s\n\nerror=%s" % (o,e)
        # There are 60 events in the test data, and the delays are
        numEvents = 60
        eventCountCmd = 'psana -m PrintEventId %s/e*-r%4.4d*.xtc | grep fiducials | grep -v "fiducials=131071" | wc' % (self.dataDir, self.run)
        evtCountOut, evtCountErr = ptl.cmdTimeOut(eventCountCmd)
        numEventsFromCmd = int(evtCountOut.split()[0])
        self.assertEqual(numEvents, numEventsFromCmd, "ran cmd=%s expected to get %d events, but got %d" % (eventCountCmd, numEvents, numEventsFromCmd))
        h5file = h5py.File(self.formatDict['h5outputFile'],'r')
        systemDelays = h5file['system/system_params/delays'][:]
        userDelays = h5file['user/G2_results_at_000060/delays'][:]
        self.assertEqual(list(systemDelays), list(userDelays), msg="in h5 output file, system and user section do not have same delays")
        expectedCounts = [numEvents - delay for delay in systemDelays]
        counts = h5file['user/G2_results_at_000060/delay_counts'][:]
        self.assertEqual(len(counts), len(expectedCounts))
        for count, expCount, delay in zip(counts, expectedCounts, systemDelays):
            pass
#            self.assertEqual(count, expCount, msg="for delay=%d expected a count=%d (given %d events) but got %d" % \
#                             (delay, expCount,  numEvents, count))

if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], '-v'])
