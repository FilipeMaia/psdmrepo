from psana import *
import numpy as np
import sys
import event_process
from mpi4py import MPI
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import math
import pylab

def array_calc(arr):
    flat = np.ravel(arr)
    return np.sum(flat)/(len(flat)-1)-flat[0]

def find_time(time):
    for i,t in enumerate(times):
        if t.time()==time:
            return i
    print 'time not found:',time




class offbyone(event_process.event_process):
    def __init__(self):
        self.output = event_process.event_process_output()
        self.reducer_rank = 0
        self.logger = logging.getLogger(__name__+'.offbyone')
        self.results = {}
        self.shotrange = 4
        self.mindroppedshots = 10 # number of data points to have to consider results "valid"
        self.output['in_report']         = 'analysis'
        self.output['in_report_title']   = 'Off By One'
        self.count162 = 0
        return

    def off_by_one_process(self,evt,eventoffset,results):
        for k in evt.keys():
            strtype = str(k.type())
            strsrc = str(k.src())
            #if 'BldDataPhaseCavity' not in strtype and 'BldDataFEEGasDetEnergy' not in strtype:
            #    continue
            if 'BldDataEBeam' in strtype:
                data = evt.get(k.type(),k.src()).ebeamCharge()
            elif 'BldDataSpectrometerV1' in strtype:
                data = evt.get(k.type(),k.src()).integral()
            elif 'BldDataFEEGasDetEnergy' in strtype:
                data = evt.get(k.type(),k.src()).f_11_ENRC()
            elif 'PNCCD.FullFrame' in strtype:
                data = array_calc(evt.get(k.type(),k.src()).data())
            elif 'Camera.Frame' in strtype:
                data = array_calc(evt.get(k.type(),k.src()).data16())
            elif 'BldDataPhaseCavity' in strtype:
                data = evt.get(k.type(),k.src()).charge1()
            elif 'IpmFex' in strtype:
                data = np.sum(evt.get(k.type(),k.src()).channel())
            else:
                continue
            if strsrc not in results:
                results[strsrc] = {}
            if eventoffset not in results[strsrc]:
                results[strsrc][eventoffset] = []
            results[strsrc][eventoffset].append(data)

    def event(self, evt):
        if 162 in self.parent.shared['evr']:
            self.count162 += 1
            for i in range(-self.shotrange,self.shotrange+1):
                if self.parent.eventN+i<0 or self.parent.eventN+i>=len(self.parent.mytimes):
                    continue
                thisevt = self.parent.thisrun.event(self.parent.mytimes[self.parent.eventN+i])
                self.off_by_one_process(thisevt,i,self.results)
        return

    def mergeresults(self, GR):
        # get all the keys:
        allkeys = []
        MR = {}
        for dd in GR:
            allkeys.extend( dd.keys() )

        for dd in GR:
            for k in set(allkeys):
                if k in MR :
                    if k in dd:
                        for ii in dd[k]:
                            if ii not in MR[k]:
                                MR[k][ii] = []
                            MR[k][ii].extend( dd[k][ii] )
                else :
                    MR[k] = dict( dd[k] )
        return MR

    def reduce(self,comm,ranks=[],reducer_rank=None,tag=None):
        self.gathered_results = comm.gather( self.results , root=reducer_rank )
        self.allcount162 = np.array([0])
        if reducer_rank is None and tag is None:
            self.allcount162[0] += self.count162
        elif reducer_rank == comm.Get_rank() and tag is not None:
            for r in ranks:
                if r == reducer_rank:
                    self.allcount162[0] += self.count162
                else :
                    self.allcount162[0] += comm.recv( source=r, tag=tag)
        elif reducer_rank != comm.Get_rank() and tag is not None:
            comm.send( np.array(self.count162), dest=reducer_rank, tag=tag )

        self.parent.comm.Reduce( 
                [np.array(self.count162), MPI.INT], 
                [self.allcount162, MPI.INT],
                op=MPI.SUM,root=reducer_rank)
        return

    def endJob(self):
        self.logger.info( 'found {:} dropped shots'.format( self.count162 ) )
        self.reduce(self.parent.comm,ranks=self.reduce_ranks,reducer_rank=self.reducer_rank, tag=77)

        if self.parent.rank == self.reducer_rank:
            self.logger.info('all dropped shots: {:}'.format( self.allcount162[0] ) )
            self.merged_results = self.mergeresults( self.gathered_results )
            self.offByOne = False
            self.output['table'] = {}
            for det in iter(self.merged_results):
                self.output['table'][det] = {}
                chisqmin = sys.float_info.max
                index = 0 # default to "success", i.e. not off-by-one
                for i in range(-self.shotrange,self.shotrange+1):
                    val = self.chisq(det,i)
                    self.logger.debug('{:} {:0.2f} {:}'.format(i,val,det))
                    if val<chisqmin:
                        chisqmin=val
                        index = i
                self.output['table'][det]['index'] = index
                self.output['table'][det]['chi2min'] = chisqmin
                self.output['table'][det]['chi2'] = -1
                self.output['table'][det]['offbyone'] = 0
                if index==0: continue    # min chisq when we exclude 0, so we're OK
                if chisqmin>10: continue # to avoid false positives, require that these shots look statistically consistent
                csq = self.chisq(det,99999)   # min chisq not at 0! compute chisq with all points to see if we have a significant deviation
                self.output['table'][det]['chi2'] = csq
                if csq>5:
                    self.logger.info( repr(det)+' off by '+repr(index)+' with chisq '+repr(csq) )
                    self.output['text'].append( '<p>'+repr(det)+' off by '+repr(index)+' with chisq '+repr(csq)+'</p>' )
                    self.offByOne = True
                    self.output['table'][det]['offbyone'] = 1
            self.output['figures'] = {}
            self.plot(self.merged_results)
            if self.allcount162[0] > self.mindroppedshots :
                self.output['text'].append( '<p>Off by One: <b>{:}</b><br/>Total dropped shots: {:0.0f}</p>'.format(self.offByOne,self.allcount162[0]) )
            else:
                self.output['text'].append( '<p>Off by One: N/A<br/>Total dropped shots: {:0.0f}</p>'.format(self.allcount162[0]) )
            self.parent.output.append(self.output)
        return

    def plot(self,results):
        totalfigs = 0
        with PdfPages('figure_offbyone.pdf') as pdf : # fix this should be pngs
            for i,det in enumerate(iter(results)):
                subplotnum=i%9
                if subplotnum==0:
                    plt.figure('Off By One')
                plt.subplot(3,3,subplotnum)
                plt.tick_params(axis='both', which='major', labelsize=6)
                plt.tick_params(axis='both', which='minor', labelsize=6)
                plt.title(det, fontsize=8)
                axes = plt.gca()
                axes.set_xlim([-self.shotrange-0.5,self.shotrange+0.5])
                for eventoffset in iter(results[det]):
                    y = results[det][eventoffset]
                    x = [eventoffset]*len(y)
                    plt.plot(x,y,'ko',alpha=0.4)
                if subplotnum in [7,8,0]:
                    plt.xlabel('event offset',fontsize=8)
                if subplotnum in [1,4,7]:
                    plt.ylabel('response [arb.]',fontsize=8)
                if subplotnum==8 or i==len(results.keys())-1:
                    pdf.savefig()
                    pylab.savefig( os.path.join( self.output_dir, 'figure_offbyone_{:}.png'.format( totalfigs ) ) )
                    self.output['figures'][totalfigs] = {}
                    self.output['figures'][totalfigs]['png'] = os.path.join( self.output_dir, 'figure_offbyone_{:}.png'.format( totalfigs ) )
                    totalfigs += 1
                    #plt.show()
                    plt.close()

    def chisq(self,det,dropGuess,**kwargs):
        points = []
        for eventoffset in iter(self.merged_results[det]):
            if eventoffset==dropGuess: continue
            vals = self.merged_results[det][eventoffset]
            self.logger.debug('vals : '+repr(vals))
            if len(vals)<5: continue
            points.append([np.mean(vals),np.var(vals)/len(vals)])
        if len(points)>1:
            return self.chisquare(points,**kwargs)
        else:
            self.logger.info(points)
            return sys.float_info.max

    def chisquare(self,points,**kwargs):
        wtdMeanVar = 1./np.sum([1/v[1] for v in points])
        wtdMean = np.sum([v[0]/v[1] for v in points])*wtdMeanVar
        chisq = np.sum([(v[0]-wtdMean)**2/v[1] for v in points])
        dof = len(points)-1
        self.logger.debug( "{:} {:} {:}".format( wtdMean,wtdMeanVar,points ) )
        return chisq/dof
