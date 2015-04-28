#--------------------------------------------------------------------------
# Description:
#   unit tests for external packages
#   
#------------------------------------------------------------------------


#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os
import StringIO
import unittest
import psana_test.psanaTestLib as ptl

DATADIR = "/reg/g/psdm/data_test/Translator"
OUTDIR = "data/psana_test"

#------------------
# Utility functions 
#------------------
#-------------------------------
#  Unit test class definition --
#-------------------------------
class ExtPkg( unittest.TestCase ) :

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

    def compareText(self,textA, textB):
        lnsA = [ln.strip() for ln in textA.split('\n') if len(ln.strip())>0]
        lnsB = [ln.strip() for ln in textB.split('\n') if len(ln.strip())>0]
        self.assertEqual(len(lnsA), len(lnsB), msg="compareText: number of lines disagree, %d != %d" % (len(lnsA),len(lnsB)))

        for lni,x,y in zip(range(len(lnsA)), lnsA, lnsB):
            self.assertEqual(x,y, msg="line %d lines not equal:\n---%s---\n---%s---" % (lni, x, y))


    def test_h5py(self):
        '''
        We test h5py by looking at the output of h5tools on 
        two complicated datasets. In particular we look at
        datasets with vlen objects in it. It is important to test
        h5py on our vlen data to see if we still need the patch we
        apply to h5py
        '''
        infile = os.path.join(DATADIR,'test_042_Translator_t1.xtc')
        assert os.path.exists(infile), "infile not found"
        outfile = os.path.join(OUTDIR, 'unit-test_h5py.h5')
        ptl.translate(infile, outfile,numEvents=0,testLabel='test_h5py',verbose=False)
        #### test that we can import h5py - however packages already imported can affect success
        successfulImport = False
        try:
            import h5py
            successfulImport = True
        except ImportError:
            self.assertTrue(successfulImport,msg="Failed to import h5py")

        #### test that we can import h5tools
        successfulImport = False
        try:
            import h5tools
            successfulImport = True
        except ImportError:
            self.assertTrue(successfulImport,msg="Failed ot import h5tools")

        expectedOutputs={}
        expectedOutputs['/Configure:0000/Ipimb::ConfigV2/XppSb2_Ipm/config']='''
 triggerCounter serialID chargeAmpRange               capacitorValue calibrationRange resetLength resetDelay chargeAmpRefVoltage calibrationVoltage diodeBias status errors calStrobeLength trigDelay trigPsDelay adcDelay
         uint64   uint64         uint16                array of enum           uint16      uint32     uint32             float32            float32   float32 uint16 uint16          uint16    uint32      uint32   uint32
              0        0              0 [c_1pF, c_1pF, c_1pF, c_1pF]                0      600000       4095                 1.0                0.0      30.0      0      0               0    250000      150000     4000
'''
        expectedOutputs['/Configure:0000/EvrData::ConfigV7/NoDetector.0:Evr.0/config']='''
 neventcodes npulses noutputs
      uint32  uint32   uint32
           5       7       12
'''
        expectedOutputs['/Configure:0000/EvrData::ConfigV7/NoDetector.0:Evr.0/eventcodes']='''
rowIdx   code isReadout isCommand isLatch reportDelay reportWidth maskTrigger maskSet maskClear desc readoutGroup releaseCode
       uint16     uint8     uint8   uint8      uint32      uint32      uint32  uint32    uint32  str       uint16      uint32
     0     67         0         1       0           0           1           0       0         0                 0           1
     1     45         0         1       0           0           1           0       0         0                 0           1
     2    140         1         0       0           0           1         127       0         0                 1           1
     3    162         0         1       0           0           1           0       0         0                 0           1
     4     41         0         0       0           0           1           0       0         0                 0           1
'''
        expectedOutputs['/Configure:0000/EvrData::ConfigV7/NoDetector.0:Evr.0/output_maps']='''
rowIdx source source_id   conn conn_id module
         enum     uint8   enum   uint8  uint8
     0  Pulse         0 UnivIO       0      0
     1  Pulse         1 UnivIO       3      0
     2  Pulse         2 UnivIO       4      0
     3  Pulse         3 UnivIO       6      0
     4  Pulse         3 UnivIO       7      0
     5  Pulse         3 UnivIO       8      0
     6  Pulse         4 UnivIO       9      0
     7  Pulse         5 UnivIO       5      0
     8  Pulse         6 UnivIO       1      0
     9  Pulse         6 UnivIO       2      0
    10  Pulse         6 UnivIO      10      0
    11  Pulse         6 UnivIO      11      0
'''
        expectedOutputs['/Configure:0000/EvrData::ConfigV7/NoDetector.0:Evr.0/pulses']='''
rowIdx pulseId polarity prescale  delay  width
        uint16   uint16   uint32 uint32 uint32
     0       0        1        1  79135   1190
     1       1        0        1  66532   1190
     2       2        0        1  59500   1190
     3       3        1        1  83300  35700
     4       4        1        1   1189    119
     5       5        0        1 110670   1190
     6       6        0        1  83300  35700
'''
        expectedOutputs['/Configure:0000/EvrData::ConfigV7/NoDetector.0:Evr.0/seq_config']='''
                                            entries (vlen)
 sync_source beam_source length cycles    delay, eventcode
        enum        enum uint32 uint32   uint32,    uint32
     Disable     Disable      0      0                    
'''
        expectedOutputs['Configure:0000/Run:0000/CalibCycle:0000/EvrData::DataV3/NoDetector.0:Evr.0/data']='''
                                fifoEvents (vlen)
rowIdx     timestampHigh, timestampLow, eventCode
                  uint32,       uint32,    uint32
     0 [(       118410,        11852,       140)]
     1 [(       118434,        11852,       140)]
'''
        f=h5py.File(outfile)
        for dsKey, expectedOutput in expectedOutputs.iteritems():
            origStdout = sys.stdout
            sys.stdout = StringIO.StringIO()
            try:
                h5tools.printds(f[dsKey])
            except Exception,e:
                sys.stdout = origStdout
                sys.stderr.write("h5tools printds call failed on ipimbConfig from h5py read\n")
                raise e

            sys.stdout.seek(0)
            printdsOutput = sys.stdout.read()
            sys.stdout = origStdout
            self.compareText(printdsOutput, expectedOutput)
        f.close()
        if self.cleanUp:
            os.unlink(outfile)

    def test_pandas(self):
        #### test that we can import pandas - however packages already imported can affect success
        successfulImport = False
        try:
            import pandas as pd
            successfulImport = True
        except ImportError:
            self.assertTrue(successfulImport,msg="Failed to import pandas")

        # from the pandas tutorial
        names = ['Bob','Jessica','Mary','John','Mel']
        births = [968, 155, 77, 578, 973]
        BabyDataSet = zip(names,births)
        df = pd.DataFrame(data = BabyDataSet, columns=['Names', 'Births'])
        self.assertEqual(df['Births'].max(),973,msg="births max is not 973")
        outfile = os.path.join(OUTDIR,'unit_test-test_pandas_df.h5')
        if os.path.exists(outfile):
            os.unlink(outfile)
        df.to_hdf(outfile, 'df')
        del df
        df_hdf = pd.read_hdf(outfile,'df')
        self.assertEqual(df_hdf['Births'].max(),973,msg="after write/read to hdf5, births max is not 973")
        if self.cleanUp:
            os.unlink(outfile)
        
    def test_pytables(self):
        #### test that we can import tables  - however packages already imported can affect success
        #### in particular h5py has already been imported
        successfulImport = False
        try:
            import tables
            successfulImport = True
        except ImportError:
            self.assertTrue(successfulImport,msg="Failed to import tables (pytables)")
        import numpy as np
        
        # do some of the pytables 3.1.1 tutorial, this is from
        # http://pytables.github.io/usersguide/tutorials.html
        class Particle(tables.IsDescription):
            name      = tables.StringCol(16)   # 16-character String
            idnumber  = tables.Int64Col()      # Signed 64-bit integer
            ADCcount  = tables.UInt16Col()     # Unsigned short integer
            TDCcount  = tables.UInt8Col()      # unsigned byte
            grid_i    = tables.Int32Col()      # 32-bit integer
            grid_j    = tables.Int32Col()      # 32-bit integer
            pressure  = tables.Float32Col()    # float  (single-precision)
            energy    = tables.Float64Col()    # double (double-precision)

        outfile = os.path.join(OUTDIR,'unit-test_pytables.h5')
        h5file = tables.open_file(outfile, mode = "w", title = "Test file")
        group = h5file.create_group("/", 'detector', 'Detector information')
        table = h5file.create_table(group, 'readout', Particle, "Readout example")
        particle = table.row
        for i in xrange(10):
            particle['name']  = 'Particle: %6d' % (i)
            particle['TDCcount'] = i % 256
            particle['ADCcount'] = (i * 256) % (1 << 16)
            particle['grid_i'] = i
            particle['grid_j'] = 10 - i
            particle['pressure'] = float(i*i)
            particle['energy'] = float(particle['pressure'] ** 4)
            particle['idnumber'] = i * (2 ** 34)
            # Insert a new particle record
            particle.append()
        table.flush()
        table = h5file.root.detector.readout
        pressures = [x['pressure'] for x in table.iterrows() if x['TDCcount'] > 3 and 20 <= x['pressure'] < 50]
        self.assertEqual(len(pressures),3,msg='tables tutorial pressure query did not have 4 values, had %d' % len(pressures))
        for val,expectedVal in zip(pressures,[25.0, 36.0, 49.0]):
            self.assertEqual(int(val),int(expectedVal), msg="a pressure value is wrong. expected=%.1f got=%.1f" % (expectedVal, val))
        del pressures
        del table
        del group
        del h5file
        if self.cleanUp:
            os.unlink(outfile)

if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], '-v'])
