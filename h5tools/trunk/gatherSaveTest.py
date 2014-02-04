import psana
import h5tools

# motor calibration
EncoderScale = 0.005 
Xoffset = 17050.
Yoffset = 1056510.
Zoffset = 1830400.

def myGatherFoo(diode, gasDet, sampleXPv, sampleYPv, sampleZPv, aHPv):
    diodeSum = diode.channel0Volts() + diode.channel1Volts() + \
               diode.channel2Volts() + diode.channel3Volts()
    aG = gasDet.f_11_ENRC()
    aX = sampleXPv.value(0) * EncoderScale - Xoffset
    aY = sampleYPv.value(0) * EncoderScale - Yoffset
    aZ = sampleZPv.value(0) * EncoderScale - Zoffset
    aH = aHPv.value(0)
    return {'aD':diodeSum,
            'aG':aG,
            'aX':aX,
            'aY':aY,
            'aZ':aZ,
            'aH':aH}

if __name__ == '__main__':
    h5output = "myoutput.h5"
    numEvents = 15
    status = 5
    print "Test of h5tools.gatherSave to write out data",
    print "from %d events to the file: '%s' with status every %d events" % (numEvents,h5output,status)
    h5tools.gatherSave(dataSource="exp=cxitut13:run=22",
                       h5OutputFileName=h5output,
                       inputOutputFunction=myGatherFoo,
                       inputs={'diode':(psana.Ipimb.DataV2, psana.Source('DetInfo(CxiDg4.0:Ipimb.0)')),
                               'gasDet':(psana.Bld.BldDataFEEGasDetEnergy, psana.Source('BldInfo(FEEGasDetEnergy)')),
                               'sampleXPv':(psana.Epics, 'CXI:SC1:MZM:08:ENCPOSITIONGET'),
                               'sampleYPv':(psana.Epics, 'CXI:SC1:MZM:09:ENCPOSITIONGET'),
                               'sampleZPv':(psana.Epics, 'CXI:SC1:MZM:10:ENCPOSITIONGET'),
                               'aHPv':(psana.Epics, 'CXI:KB1:MMS:07.RBV') },                       
                       overwrite=True,
                       status=status,
                       numEvents=numEvents)
    dataArray, timeArray, posArray = h5tools.H5ReadDataTimePos(h5output)
    print "-- Object representation for first rows of numpy arrays read back from hdf5 file:--"
    print "** data[0:3] **"
    print "%r" % dataArray[0:3]
    print
    print "** time[0:3] **"
    print "%r" % timeArray[0:3]
    print
    print "** pos[0:3] **"
    print "%r" % posArray[0:3]

