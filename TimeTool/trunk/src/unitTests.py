import os
import unittest
import psana

class Analyze(unittest.TestCase):
    def setUp(self):
        self.longMessage = True
        DATADIR = '/reg/g/psdm/data_test/multifile/test_014_sxri0214'
        assert os.path.exists(DATADIR), "testing datadir=%s doesn't exist" % DATADIR
        self.datasource = 'exp=sxri0214:run=158:dir=%s' % DATADIR 

        self.EVR_BYKICK = 162
        self.TT_PUTKEY = 'TTANA'
        self.psanaOptions = {
            ########## psana configuration #################
            'psana.modules':'TimeTool.Analyze',

            ########## TimeTool.Analyze configuration #######
            #  Key for fetching timetool camera image
            'TimeTool.Analyze.get_key':'TSS_OPAL',
            #  Results are written to <put_key>
            'TimeTool.Analyze.put_key':self.TT_PUTKEY,
            #  Indicate absence of beam for updating reference
            'TimeTool.Analyze.eventcode_nobeam':self.EVR_BYKICK,
            #  Indicate events to skip (no laser, for example)
            'TimeTool.Analyze.eventcode_skip':0,
            #  Polynomial coefficients for position_time calculation
            'TimeTool.Analyze.calib_poly':'0 1 0',
            #  Project onto X axis?
            'TimeTool.Analyze.projectX':True,
            #  Minimum required bin value of projected data
            'TimeTool.Analyze.proj_cut':0,
            #  ROI (x) for signal
            'TimeTool.Analyze.sig_roi_x':'0 1023',
            #  ROI (y) for signal
            'TimeTool.Analyze.sig_roi_y':'408 920',
            #  ROI (x) for sideband
            'TimeTool.Analyze.sb_roi_x':'' ,
            #  ROI (y) for sideband
            'TimeTool.Analyze.sb_roi_y':'', 
            #  Rolling average convergence factor (1/Nevents)
            'TimeTool.Analyze.sb_avg_fraction':0.05,
            #  Rolling average convergence factor (1/Nevents)
            'TimeTool.Analyze.ref_avg_fraction':1.0,
            #  Read weights from a text file
            'TimeTool.Analyze.weights_file':'',
            #  Indicate presence of beam from IpmFexV1::sum() [monochromator]
            'TimeTool.Analyze.ipm_get_key':'',
            # 'TimeTool.Analyze.ipm_beam_threshold':'',
            #  Load initial reference from file
            'TimeTool.Analyze.ref_load':'',
            #  Save final reference to file
            'TimeTool.Analyze.ref_store':'timetool.ref',
            #  Generate histograms for initial events, dumped to root file (if non zero)
            'TimeTool.Analyze.dump':0,
            #  Filter weights
            'TimeTool.Analyze.weights':'0.00940119 -0.00359135 -0.01681714 -0.03046231 -0.04553042 -0.06090473 -0.07645332 -0.09188818 -0.10765874 -0.1158105  -0.10755824 -0.09916765 -0.09032289 -0.08058788 -0.0705904  -0.06022352 -0.05040479 -0.04144206 -0.03426838 -0.02688114 -0.0215419  -0.01685951 -0.01215143 -0.00853327 -0.00563934 -0.00109415  0.00262359  0.00584445  0.00910484  0.01416929  0.0184887   0.02284319  0.02976289  0.03677404  0.04431778  0.05415214  0.06436626  0.07429347  0.08364909  0.09269116  0.10163601  0.10940983  0.10899065  0.10079016  0.08416471  0.06855799  0.05286105  0.03735241  0.02294275  0.00853613',
        }
        
    @unittest.skip("interactive plotting example")
    def test_plot(self):
        psana.setConfigFile("")
        self.psanaOptions['TimeTool.Analyze.eventdump']=True
        self.psanaOptions['psana.modules'] += ' TimeTool.PlotAnalyze'
        psana.setOptions(self.psanaOptions)
        ds = psana.DataSource(self.datasource)
        self.runThroughDataSourceAndCheckAnswers(ds)

    def test_basic(self):
        psana.setConfigFile("")
        psana.setOptions(self.psanaOptions)
        ds = psana.DataSource(self.datasource)
        # below we record the timetool answers we saw when writing this test. The key is an event index. If there is no key, then the
        # time tool produced no result (for events 0-16 and event 18. If psalg or the TimeTool changes, it is reaonsable that these 
        # values will change, in which case the test should be modified. To modify the test, the line that prints these dictionaries
        # is commented out in the test below
        self.runThroughDataSourceAndCheckAnswers(ds)

    def runThroughDataSourceAndCheckAnswers(self,ds):
        timeToolAnswers = {
            17:{'amplitude':0.0098, 'nxt_amplitude':0.0049, 'position_fwhm':21.0986, 'ref_amplitude':525459.00, 'position_pixel':288.04},
            19:{'amplitude':0.0054, 'nxt_amplitude':0.0026, 'position_fwhm':30.0244, 'ref_amplitude':507888.00, 'position_pixel':352.72},
            20:{'amplitude':0.0065, 'nxt_amplitude':0.0034, 'position_fwhm':31.4669, 'ref_amplitude':487428.00, 'position_pixel':370.91},
            21:{'amplitude':0.0066, 'nxt_amplitude':0.0026, 'position_fwhm':31.1662, 'ref_amplitude':500426.00, 'position_pixel':354.60},
            22:{'amplitude':0.0120, 'nxt_amplitude':0.0092, 'position_fwhm':14.4309, 'ref_amplitude':507888.00, 'position_pixel':349.70},
            23:{'amplitude':0.0087, 'nxt_amplitude':0.0028, 'position_fwhm':25.8615, 'ref_amplitude':502523.00, 'position_pixel':330.24},
            24:{'amplitude':0.0065, 'nxt_amplitude':0.0033, 'position_fwhm':35.5341, 'ref_amplitude':509406.00, 'position_pixel':343.31},
            25:{'amplitude':0.0070, 'nxt_amplitude':0.0043, 'position_fwhm':57.3005, 'ref_amplitude':502031.00, 'position_pixel':349.10},
            26:{'amplitude':0.0091, 'nxt_amplitude':0.0065, 'position_fwhm':30.8810, 'ref_amplitude':511500.00, 'position_pixel':338.47},
            27:{'amplitude':0.0087, 'nxt_amplitude':0.0073, 'position_fwhm':39.7903, 'ref_amplitude':507888.00, 'position_pixel':351.18},
            28:{'amplitude':0.0150, 'nxt_amplitude':0.0143, 'position_fwhm':15.5931, 'ref_amplitude':530145.00, 'position_pixel':591.59},
            29:{'amplitude':0.0141, 'nxt_amplitude':0.0100, 'position_fwhm':18.8107, 'ref_amplitude':519140.00, 'position_pixel':600.09},
            30:{'amplitude':0.0123, 'nxt_amplitude':0.0116, 'position_fwhm':21.6014, 'ref_amplitude':516674.00, 'position_pixel':601.06},
            31:{'amplitude':0.0181, 'nxt_amplitude':0.0147, 'position_fwhm':30.1255, 'ref_amplitude':511500.00, 'position_pixel':339.36},
            32:{'amplitude':0.0167, 'nxt_amplitude':0.0161, 'position_fwhm':23.3140, 'ref_amplitude':507377.00, 'position_pixel':56.24},
            33:{'amplitude':0.0191, 'nxt_amplitude':0.0180, 'position_fwhm':23.7252, 'ref_amplitude':507377.00, 'position_pixel':56.40},
            34:{'amplitude':0.0207, 'nxt_amplitude':0.0202, 'position_fwhm':27.9930, 'ref_amplitude':508661.00, 'position_pixel':347.09},
            35:{'amplitude':0.0254, 'nxt_amplitude':0.0235, 'position_fwhm':24.9222, 'ref_amplitude':509830.00, 'position_pixel':343.82},
        }

        for idx,evt in enumerate(ds.events()):
            ttData = evt.get(psana.TimeTool.DataV2, self.TT_PUTKEY)
            if idx not in timeToolAnswers:
                self.assertIsNone(ttData, msg="idx=%d. TimeTool produced answer but we did not expect one" % idx)
                continue
            self.assertIsNotNone(ttData, msg="idx=%d. TimeTool produced no answer but expect one" % idx)
            answers = timeToolAnswers[idx]
            self.assertAlmostEqual(answers['amplitude'], ttData.amplitude(), places=3, msg="event=%d amplitudes" % idx)
            self.assertAlmostEqual(answers['nxt_amplitude'], ttData.nxt_amplitude(), places=3, msg="event=%d nxt_amplitudes" % idx)
            self.assertAlmostEqual(answers['position_fwhm'], ttData.position_fwhm(), places=1, msg="event=%d aposition_fwhm" % idx)
            self.assertAlmostEqual(answers['ref_amplitude'], ttData.ref_amplitude(), places=0, msg="event=%d ref_amplitudes" % idx)
            self.assertAlmostEqual(answers['position_pixel'], ttData.position_pixel(), places=0, msg="event=%d position_pixel" % idx)
#            print "%d:{'amplitude':%.4f, 'nxt_amplitude':%.4f, 'position_fwhm':%.4f, 'ref_amplitude':%.2f, 'position_pixel':%.2f}," % \
#                (idx,ttData.amplitude(), ttData.nxt_amplitude(), ttData.position_fwhm(), ttData.ref_amplitude(), ttData.position_pixel())
            
