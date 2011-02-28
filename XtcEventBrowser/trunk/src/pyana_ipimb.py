#
# ipimb.py: plot beamline data
#
# 

import numpy as np
import matplotlib.pyplot as plt

from   pypdsdata import xtc


# analysis class declaration
class  pyana_ipimb ( object ) :
    
    def __init__ ( self, ipimb_addresses = None ) :
        # initialize data

        if ipimb_addresses is None :
            print "Error! You've called pyana_ipimb without specifying an ipimb address"
            
        self.ipimb_addresses = ipimb_addresses.split(" ")
        print "pyana_ipimb, %d sources: " % len(self.ipimb_addresses)
        for sources in self.ipimb_addresses :
            print "  ", sources
                    
        self.fex_sum = {}
        self.fex_channels = {}
        self.fex_position = {}
        for addr in self.ipimb_addresses :
            self.fex_sum[addr] = list()
            self.fex_channels[addr] = list()
            self.fex_position[addr] = list()


    def beginjob ( self, evt, env ) : 
        pass

    def event ( self, evt, env ) :

        # IPM diagnostics, for saturation and low count filtering
        for addr in self.ipimb_addresses :

            # raw data
            ipmRaw = evt.get(xtc.TypeId.Type.Id_IpimbData, addr )
            if ipmRaw :
                pass
            else :
                print "No object of type %s found" % self.ipm_addr

            # feature-extracted data
            ipmFex = evt.get(xtc.TypeId.Type.Id_IpmFex, addr )
            print type(ipmFex)
            if ipmFex :
                self.fex_sum[addr].append( ipmFex.sum )
                self.fex_channels[addr].append( ipmFex.channel )
                self.fex_position[addr].append( [ipmFex.xpos, ipmFex.ypos] )
            else :
                print "No object of type %s found" % self.ipm_addr


        
                
    def endjob( self, env ) :

        ncols = 3
        nrows = len(self.ipimb_addresses)
        print "Will now start to produce plots from %d IPMs in %d rows and %d columns" % \
              (len(self.ipimb_addresses), nrows, ncols)

        fig = plt.figure( figsize=(10*ncols/2,10*nrows/2) )

        i = 0
        for addr in self.ipimb_addresses :

            i+=1
            ax1 = fig.add_subplot(nrows, ncols, i)
            array = np.float_(self.fex_sum[addr])
            plt.hist(array, 60)
            plt.title(addr)
            plt.xlabel('Sum of channels',horizontalalignment='left') # the other right
            
            i+=1
            ax2 = fig.add_subplot(nrows, ncols, i)
            array = np.float_(self.fex_channels[addr])
            xaxis = np.arange( 0, len(self.fex_channels[addr]) )
            plt.plot(xaxis, array[:,0],xaxis, array[:,1],xaxis, array[:,2],xaxis, array[:,3])
            plt.title(addr)
            plt.xlabel('Channels',horizontalalignment='left') # the other right
            leg = ax2.legend(('ch0','ch1','ch2','ch3'),'upper center')
            
            i+=1
            ax3 = fig.add_subplot(nrows, ncols, i)
            array2 = np.float_(self.fex_position[addr])

            plt.scatter(array2[:,0],array2[:,1])
            plt.title(addr)
            plt.xlabel('Beam position X',horizontalalignment='left')
            plt.ylabel('Beam position Y',horizontalalignment='left')



        plt.show()
