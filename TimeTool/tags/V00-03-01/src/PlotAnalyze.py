import psana
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

def add_roi(plt, roilo_roihi_pdim, pltarg, name):
    roilo = list(roilo_roihi_pdim[0:2])
    roihi = list(roilo_roihi_pdim[2:4])
    pdim = roilo_roihi_pdim[4]
    ymin,xmin = roilo
    ymax,xmax = roihi
    print "roi for %s: xlim=[%d,%d] yim=[%d,%d] pdim=%d" % (name, xmin, xmax, ymin,ymax, pdim)
    plt.plot([xmin,xmin,xmax,xmax,xmin],
             [ymin,ymax,ymax,ymin,ymin], pltarg, label=name+'_roi')

class PlotAnalyze(object):
    def __init__(self):
        self.maxPrintIfNoTimeToolEventDump=20
        self.numPrintsNoTimeToolEventDump=0
        self.validReturnReasons = ["no_evr_data", "no_laser", "no_timetool_data", "no_frame_data",
                                   "8bit_frame_data", "fails_proj_cut", "nobeam", "no_reference",
                                   "analyze_event_but_nobeam_for_event", "analyze_event_nobeam",
                                   "success", "no_parab_fit", "no_peaks"]
                                   
    def beginjob(self,evt,env):
        pass

    def beginrun(self,evt,env):
        self.frameSource = psana.Source(self.configStr("tt_get_key", 'TSS_OPAL'))
        self.ttPutKey = self.configStr("tt_put_key", 'TTANA')
        self.figNumber = self.configInt("fignumber",11)
        self.pause = self.configBool('pause',True)

        self.returnReasonKey = self.ttPutKey + "_return"
        self.noLaserNoBeamKey = self.ttPutKey + "_nolaser_nobeam"

        self.sigKey = self.ttPutKey + "_sig"
        self.sbKey = self.ttPutKey + "_sb"
        self.refKey = self.ttPutKey + "_ref"

        self.sigKeyRoiLoRoiHiPdim = self.ttPutKey + "_sig_roilo_roihi_pdim"
        self.sbKeyRoiLoRoiHiPdim = self.ttPutKey + "_sb_roilo_roihi_pdim"
        self.refKeyRoiLoRoiHiPdim = self.ttPutKey + "_ref_roilo_roihi_pdim"

        self.sbAvgKey = self.ttPutKey + "_sb_avg"
        self.sbcKey = self.ttPutKey + "_sb_commonMode"
        self.refAvgKey = self.ttPutKey + "_ref_avg"
        self.sigdKey = self.ttPutKey + "_sigd"  # double version of sb
        self.qwfKey = self.ttPutKey + "_qwf"  # double version of sb
        
    def begincalibcycle(self,evt,env):
        pass

    def event(self,evt,env):
        ttReturnReason = evt.get(str, self.ttPutKey + '_return')
        if ttReturnReason is None:
            if self.numPrintsNoTimeToolEventDump < self.maxPrintIfNoTimeToolEventDump:
                sys.stderr.write("Warning: PlotAnalyze: no TimeTool event dump information found." + \
                                 "Set eventdump=1 in TimeTool.Analyze configuration and set tt_get_key and tt_put_key of " + \
                                 "this module to match get_key and put_key of the TimeTool.Analyze module.")
                self.numPrintsNoTimeToolEventDump += 1
            return

        assert ttReturnReason in self.validReturnReasons, "return reason=%s not in expected return reasons: %s" % \
            (ttReturnReason, ','.join(self.validReturnReasons))

        if ttReturnReason in ['no_evr_data', 'no_laser', 'no_timetool_data', 'no_frame_data', '8bit_frame_data', \
                              'no_reference','analyze_event_nobeam', 'analyze_event_but_nobeam_for_event']: 
            return

        frameData = evt.get(psana.Camera.FrameV1, self.frameSource).data16()

        sig = evt.get(psana.ndarray_int32_1, self.sigKey)
        sb = evt.get(psana.ndarray_int32_1, self.sbKey)
        ref = evt.get(psana.ndarray_int32_1, self.refKey)

        sig_roilo_roihi_pdim = evt.get(psana.ndarray_uint32_1, self.sigKeyRoiLoRoiHiPdim)
        sb_roilo_roihi_pdim = evt.get(psana.ndarray_uint32_1, self.sbKeyRoiLoRoiHiPdim)
        ref_roilo_roihi_pdim = evt.get(psana.ndarray_uint32_1, self.refKeyRoiLoRoiHiPdim)

        sb_avg =  evt.get(psana.ndarray_float64_1, self.sbAvgKey)
        sbc =  evt.get(psana.ndarray_float64_1, self.sbcKey)

        ref_avg = evt.get(psana.ndarray_float64_1, self.refAvgKey)
        
        sigd = evt.get(psana.ndarray_float64_1, self.sigdKey)

        qwf = evt.get(psana.ndarray_float64_1, self.qwfKey)

        if ttReturnReason == "nobeam":
            # maybe reference updated
            pass
        elif ttReturnReason == 'success':
            pass
        elif ttReturnReason == 'no_parab_fit':
            pass
        elif ttReturnReason == 'no_peaks':
            pass

        ttData = evt.get(psana.TimeTool.DataV2, self.ttPutKey)

        plt.figure(self.figNumber)
        plt.clf()
        plt.imshow(frameData)
        plt.hold(True)
        if sig_roilo_roihi_pdim is not None:
            add_roi(plt, sig_roilo_roihi_pdim, 'r', 'sig')
        else:
            print "no signal roi"
        if sb_roilo_roihi_pdim is not None:
            add_roi(plt, sb_roilo_roihi_pdim, 'c', 'sb')
        if ref_roilo_roihi_pdim is not None:
            add_roi(plt, ref_roilo_roihi_pdim, 'y', 'ref')
        plt.legend()
        plt.title("frame data: " + ttReturnReason)
        plt.draw()

        plt.figure(self.figNumber+1,(8,12))
        NUMPLOTS=5
        plt.clf()
        plt.subplot(NUMPLOTS,1,1)
        if sig is not None: 
            plt.plot(sig)
        plt.title('raw signal')
        plt.subplot(NUMPLOTS,1,2)
        if sigd is not None:
            plt.plot(sigd)
        plt.title('sigd (normalized, divided by reference)')

        plt.subplot(NUMPLOTS,1,3)
        if sb is not None:
            plt.plot(sb, label='sb')
        plt.hold(True)
        if sb_avg is not None:
            plt.plot(sb_avg, label='sb_avg')
        if sbc is not None:
            plt.plot(sbc, label='sbc')
        plt.legend()
        plt.title('sb')

        plt.subplot(NUMPLOTS,1,4)
        if ref is not None:
            plt.plot(ref, label='ref')
        plt.hold(True)
        if ref_avg is not None:
            plt.plot(ref_avg, label='ref_avg')
        plt.legend()
        plt.title('ref')

        plt.subplot(NUMPLOTS,1,5)
        if qwf is not None:
            plt.plot(qwf, label='qwf')
            plt.hold(True)
            if ttData is not None:
                pixpos = ttData.position_pixel()
                plt.plot([pixpos,pixpos],
                         [np.min(qwf),np.max(qwf)],
                         label='tt pix pos')
        plt.legend()
        plt.title('qwf')
        plt.draw()

        if self.pause:
            raw_input("hit enter to continue: ")
