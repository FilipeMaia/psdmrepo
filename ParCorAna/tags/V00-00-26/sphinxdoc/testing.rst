
.. _testing:

################
 Testing
################

The framework provides support for testing the correlation calculation
and viewer publishing. The user provides an alternative calculation/viewer publishing
function. The results of these two calculations are then compared.
Often it is easiest to re-use the same viewer publishing and just implement a 
different workerCalc function.

When comparing results of the two calculations, it is important to make sure
they get the same input. The UserG2 module can modify the input to the 
correlation calculations. This happens through the server callbacks, and 
the worker callback workerAdjustData. When the framework runs the 
alternative testing calculation, it goes through all the data, calling these 
callbacks at the appropriate times. All of the data is passed
to the alternative testing calculation after the data has been
processed.

That is, the alternative calcuation is not something that deals with
the stored data through the WorkerData class in the framework. It
simply receives all the data at once, with sorted timecodes and counters.
It can implement a simpler, slower, more straightforward calculation.
We will see an example below in the alttest_ section.

For testing purposes, it is not neccessary to compare results on all the
pixels in the detector. Users should produce a testing mask file that 
identifies a small number of pixels, say 10-100, for testing. 

Next we go over the steps for testing.


Testing Steps
==================================================

* override the function *calcAndPublishForTestAlt* in the UserG2 class, example below.
* set *system_params['testNumEvents']* in the config file to the number of events to test with
* set *system_params['testMaskNdarrayCoords']* in the config file to the testing mask file
* set *system_params['h5output']* in the config file to the output file when you run the framework on the testing mask and testing number of events above
* set *system_params['testh5output']* in the config file to the output file for the alternative testing calculation
* set *system_params['update']* to 0 to just get results saved after going through the test number of events
* set *user_params['psmon_plot']* to False

Finally, one does::

  mpiexec -n 4 parCorAnaDriver -c myconfig.py --test_main

This creates the *h5output* argument using the testing mask and running on the testNumEventsTo run the usual calculation with the testing parameters. One could run this in on the batch system. Then::

  parCorAnaDriver -c myconfig.py --test_alt

to run the alternative calculation. Note that the alternative calculation is *not* run in MPI. 

Finally, there is a tool to compare the results. One can compare the two h5 files with the command::

  parCorAnaDriver -c myconfig.py --cmp

which compares the two h5output files defined in the config file. This runs a separate tool - 
cmpParCorAnaH5OutputPy which is part of the ParCorAna package.

As mentioned in the tutorial, parCorAnaDriver in the compare mode identifies the two h5output files to compare from the config file. If these filenames have the %C and %T options
in them, the driver will not get the correct filenames for the cmpParCorAnaH5OutputPy tool.


.. _alttest:

G2Common Example
=================
This function is as follows::

    def calcAndPublishForTestAltHelper(self,sortedEventIds, sortedData, h5GroupUser, startIdx):
        '''This implements the alternative test calculation in a way that is suitable for the sub classes
        
        The Incremental will always go through the data from the start, so startingIndex should
        be 0. However the atEnd and Windowed will only work through the last numEvents of 
        data, so startingIndex should be that far back.
        '''
        G2 = {}
        IP = {}
        IF = {}
        counts = {}
        assert len(sortedData.shape)==2
        assert sortedData.shape[0]==len(sortedEventIds)
        assert startIdx < len(sortedEventIds), "startIdx > last index into data"
        numPixels = sortedData.shape[1]
        for delay in self.delays:
            counts[delay]=0
            G2[delay]=np.zeros(numPixels)
            IP[delay]=np.zeros(numPixels)
            IF[delay]=np.zeros(numPixels)

        self.mp.logInfo("UserG2.calcAndPublishForTestAltHelper: starting double loop")

        for idxA, eventIdA in enumerate(sortedEventIds):
            if idxA < startIdx: continue
            counterA = eventIdA['counter']
            for idxB in range(idxA+1,len(sortedEventIds)):
                counterB = sortedEventIds[idxB]['counter']
                delay = counterB - counterA
                if delay == 0:
                    print "warning: unexpected - same counter twice in data - idxA=%d, idxB=%d" % (idxA, idxB)
                    continue
                if delay not in self.delays: 
                    continue
                counts[delay] += 1
                dataA = sortedData[idxA,:]
                dataB = sortedData[idxB,:]
                G2[delay] += dataA*dataB
                IP[delay] += dataA
                IF[delay] += dataB

        self.mp.logInfo("calcAndPublishForTestAlt: finished double loop")

        name2delay2ndarray = {}
        for nm,delay2masked in zip(['G2','IF','IP'],[G2,IF,IP]):
            name2delay2ndarray[nm] = {}
            for delay,masked in delay2masked.iteritems():
                name2delay2ndarray[nm][delay]=np.zeros(self.maskNdarrayCoords.shape, np.float64)
                name2delay2ndarray[nm][delay][self.maskNdarrayCoords] = masked[:]
            
        saturatedElements = np.zeros(self.maskNdarrayCoords.shape, np.int8)
        saturatedElements[self.maskNdarrayCoords] = self.saturatedElements[:]

        lastEventTime = {'sec':sortedEventIds[-1]['sec'],
                         'nsec':sortedEventIds[-1]['nsec'],
                         'fiducials':sortedEventIds[-1]['fiducials'],
                         'counter':sortedEventIds[-1]['counter']}

        countsForViewerPublish = np.zeros(len(counts),np.int)
        for idx, delay in enumerate(self.delays):
            countsForViewerPublish[idx]=counts[delay]
        self.viewerPublish(countsForViewerPublish, lastEventTime, 
                           name2delay2ndarray, saturatedElements, h5GroupUser)


  
 
