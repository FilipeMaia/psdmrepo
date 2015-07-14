#--------------------------------------------------------------------------
# Description:
#   Test script for ParCorAna
#   
#------------------------------------------------------------------------


#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
#import os
#import stat
#import tempfile
import unittest
#import subprocess as sb
#import collections
#import math
import numpy as np
#import glob
#-----------------------------
# Imports for other modules --
#-----------------------------
#import psana
#import h5py
#import psana_test.psanaTestLib as ptl

import ParCorAna as corAna

class WorkerDataNoCallback( unittest.TestCase ):
    '''Test WorkerData without a callback. 
    '''
    def setUp(self) :
    	""" 
    	Method called to prepare the test fixture. This is called immediately 
    	before calling the test method; any exception raised by this method 
    	will be considered an error rather than a test failure.  

        Setup the same WorkerData for all the tests - stores a few times, does
        not use a callback
    	"""
        self.longMessage = True
        logger = corAna.makeLogger(isTestMode=True,isMaster=True,isViewer=True,isServer=True,rank=0)
        isFirstWorker = True
        self.numTimes = 5
        numDataPointsThisWorker = 1
        
        self.workerData = corAna.WorkerData(logger, isFirstWorker, self.numTimes, 
                                            numDataPointsThisWorker, addRemoveCallbackObject = None)

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
        
    def test_addDataInOrder(self):
        '''This tests the implementation of WorkerData. I.e, that the _timesXInds grows and that once
        we fill X, we drop the earliest time.
        '''

        times = range(4*self.numTimes)  # wrap around several times
        def mkArray(x):
            a=np.zeros(1,dtype=np.float)
            a[0]=x
            return a
        datas = [mkArray(t) for t in times]
        for tm,data in zip(times, datas):
            self.workerData.addData(tm, data)
            xIdx = self.workerData.tm2idx(tm)
            self.assertEqual(xIdx, tm % self.numTimes, \
                             "tm2idx(%d)=%r != %d %% %d == %d" % (tm, self.workerData.tm2idx(tm), 
                                                                  tm, self.numTimes, tm%self.numTimes))
            self.assertAlmostEqual(self.workerData.X[xIdx,0], tm, \
                                   msg="data stored for tm=%d at idx=%d wrong" % (tm,xIdx))

        # test timesDataIndexes
        answers = zip(range(15,20),range(5))
        for tm,idx in self.workerData.timesDataIndexes():
            tmAns,idxAns = answers.pop(0)
            self.assertEqual(tm,tmAns)
            self.assertEqual(idx,idxAns)
            xVal = self.workerData.X[idx,0]
            self.assertEqual(xVal,tm)

        # test tm2idx
        self.assertIsNone(self.workerData.tm2idx(0))
        self.assertIsNone(self.workerData.tm2idx(10))
        self.assertIsNone(self.workerData.tm2idx(14))
        self.assertEqual(self.workerData.tm2idx(15),0)
        self.assertEqual(self.workerData.tm2idx(16),1)
        self.assertEqual(self.workerData.tm2idx(19),4)

    def test_empty(self):
        initialTSize = self.workerData._timesXInds.shape[0]

        self.assertTrue(self.workerData.empty())
        self.assertEqual(self.workerData._timeAfterEndIdx,0)
        self.assertEqual(self.workerData._timeStartIdx,0)
        self.assertEqual(self.workerData._timesXInds.shape[0], initialTSize)

        self.workerData._growTimesIfNeeded()
        
        self.assertTrue(self.workerData.empty())
        self.assertEqual(self.workerData._timeAfterEndIdx,0)
        self.assertEqual(self.workerData._timeStartIdx,0)
        self.assertEqual(self.workerData._timesXInds.shape[0], initialTSize)

        self.assertRaises(AssertionError, corAna.WorkerData.minTimeForStoredData, self.workerData)
        self.assertRaises(AssertionError, corAna.WorkerData.maxTimeForStoredData, self.workerData)

        self.assertEqual(len(self.workerData.timesForStoredData()),0)

    def test_addDataOutOfOrder(self):
        tms = [20,19,10,13,18]

        def mkArrPlus10(x):
            ar = np.zeros(1)
            ar[0]=x+10
            return ar

        assert self.workerData.X.shape[0] == len(tms)
        for idx,tm in enumerate(tms):
            self.assertTrue(self.workerData.addData(tm,mkArrPlus10(tm)), msg="Couldn't add data")
            xIdx = self.workerData.tm2idx(tm)
            self.assertIsNotNone(xIdx)
            self.assertEqual(xIdx,idx)
            val = self.workerData.X[xIdx,0]
            self.assertAlmostEqual(10+tm,val)
            if idx <4:
                self.assertFalse(self.workerData.filledX())

        self.assertTrue(self.workerData.filledX())

        # we shouldn't be able to add in the earliest tm
        firstTm = min(tms)
        beforeXs = [self.workerData.X[idx,0] for tm,idx in self.workerData.timesDataIndexes()]
        self.assertFalse(self.workerData.addData(firstTm, mkArrPlus10(max(beforeXs)+20)), msg="added first tm=%d" % firstTm)
        afterXs = [self.workerData.X[idx,0] for tm,idx in self.workerData.timesDataIndexes()]

        self.assertItemsEqual(beforeXs, afterXs)
        
        # we should also not be able to add in a time that is less then the earliest time
        self.assertFalse(self.workerData.addData(firstTm-1, mkArrPlus10(max(beforeXs)+20)), msg="added earlier time when filled")
        afterXs = [self.workerData.X[idx,0] for tm,idx in self.workerData.timesDataIndexes()]

        self.assertItemsEqual(beforeXs, afterXs)

        # and we should not be able to add in something we have seen
        self.assertFalse(self.workerData.addData(tms[0], mkArrPlus10(max(beforeXs)+20)), msg="added duplicated data")
        afterXs = [self.workerData.X[idx,0] for tm,idx in self.workerData.timesDataIndexes()]

        self.assertItemsEqual(beforeXs, afterXs)
        self.assertTrue(self.workerData.filledX())

        # now we will add something in the middle that makes us wrap
        # but first save where we know the xData should go
        xIndForCurrentMinimumTime = self.workerData.tm2idx(min(tms))
        self.workerData.addData(11, mkArrPlus10(11+20))
        xIdx = self.workerData.tm2idx(11)        
        self.assertEqual(xIdx,xIndForCurrentMinimumTime)
        self.assertAlmostEqual(self.workerData.X[xIdx,0], 11+20+10)
        afterXs = [self.workerData.X[idx,0] for tm,idx in self.workerData.timesDataIndexes()]
        # the minumum time, 10 should be removed
        self.assertTrue(10+10 in beforeXs)
        self.assertFalse(10+10 in afterXs)
        self.assertIsNone(self.workerData.tm2idx(10))
        self.assertTrue(self.workerData.filledX())

class WorkerDataCallback( unittest.TestCase ) :

    def setUp(self) :
    	""" 
    	Method called to prepare the test fixture. This is called immediately 
    	before calling the test method; any exception raised by this method 
    	will be considered an error rather than a test failure.  
    	"""
        self.longMessage = True
        logger = corAna.makeLogger(isTestMode=True,isMaster=True,isViewer=True,isServer=True,rank=0)
        isFirstWorker = True
        self.numTimes = 5
        numDataPointsThisWorker = 1

        self.callbacks = {'remove':[],
                          'add':[],
                          'adjust':[]}

        class CallBack(object):
            def __init__(self, callbacks):
                self.callbacks = callbacks
                self.numpyArrayType = type(np.zeros(13))

            def workerBeforeDataRemove(self, tm, dataIdx, wd):
                assert isinstance(tm, int)
                assert isinstance(dataIdx, int)
                self.callbacks['remove'].append((tm,wd.X[dataIdx,0]))

            def workerAdjustData(self, data):
                assert isinstance(data, self.numpyArrayType)
                self.callbacks['adjust'].append(data[0])

            def workerAfterDataInsert(self, tm, dataIdx, wd):
                assert isinstance(tm, int)
                assert isinstance(dataIdx, int)
                self.callbacks['add'].append((tm,wd.X[dataIdx,0]))
                
        self.workerData = corAna.WorkerData(logger, isFirstWorker, 
                                            self.numTimes, 
                                            numDataPointsThisWorker, 
                                            addRemoveCallbackObject = CallBack(self.callbacks))


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

    def test_addDataInOrder(self):
        times = range(20)
        def mkArrayPlus10(x):
            a=np.zeros(1, dtype=np.float)
            a[0]=10+x
            return a
        for tm in times:
            self.workerData.addData(tm,mkArrayPlus10(tm))

        self.assertEqual(len(self.callbacks['adjust']),20)
        self.assertEqual(len(self.callbacks['add']),20)
        self.assertEqual(len(self.callbacks['remove']),15)

        adjustAnswer = map(float,range(10,30))
        addAnswer = [(tm,float(10+tm)) for tm in range(20)]
        removeAnswer = [(tm,float(10+tm)) for tm in range(15)]

        for idx, ans, val in zip(range(20), adjustAnswer, self.callbacks['adjust']):
            self.assertAlmostEqual(ans, val, msg='adjust callback, entry=%d. ans != val' % (idx,))

        for idx, ans, val in zip(range(20), addAnswer, self.callbacks['add']):
            self.assertAlmostEqual(ans[1], val[1], msg='add callback, entry=%d, x data, ans != val' % (idx,))
            self.assertEqual(ans[0], val[0], msg='add callback, entry=%d, time data, ans != val' % (idx,))

        for idx, ans, val in zip(range(15), removeAnswer, self.callbacks['remove']):
            self.assertAlmostEqual(ans[1], val[1], msg='remove callback, entry=%d, expected x data, ans != val' % (idx,))
            self.assertEqual(ans[0], val[0], msg='remove callback, entry=%d, expected tm data, ans != val' % (idx,))

    def test_addDataOutOfOrder(self):
        times = range(20)
        def mkArrayPlus10(x):
            a=np.zeros(1, dtype=np.float)
            a[0]=10+x
            return a
        for tm in times:
            self.workerData.addData(tm,mkArrayPlus10(tm))

        self.assertEqual(len(self.callbacks['adjust']),20)
        self.assertEqual(len(self.callbacks['add']),20)
        self.assertEqual(len(self.callbacks['remove']),15)

        adjustAnswer = map(float,range(10,30))
        addAnswer = [(tm,float(10+tm)) for tm in range(20)]
        removeAnswer = [(tm,float(10+tm)) for tm in range(15)]

        for idx, ans, val in zip(range(20), adjustAnswer, self.callbacks['adjust']):
            self.assertAlmostEqual(ans, val, msg='adjust callback, entry=%d. ans != val' % (idx,))

        for idx, ans, val in zip(range(20), addAnswer, self.callbacks['add']):
            self.assertAlmostEqual(ans[1], val[1], msg='add callback, entry=%d, x data, ans != val' % (idx,))
            self.assertEqual(ans[0], val[0], msg='add callback, entry=%d, time data, ans != val' % (idx,))

        for idx, ans, val in zip(range(15), removeAnswer, self.callbacks['remove']):
            self.assertAlmostEqual(ans[1], val[1], msg='remove callback, entry=%d, expected x data, ans != val' % (idx,))
            self.assertEqual(ans[0], val[0], msg='remove callback, entry=%d, expected tm data, ans != val' % (idx,))


#################################################
######### TEST CORRELATION CALCULATION ##########

def makePairsAnswer(times, data, delays):
    '''Testing tool to produce the pairs for the given delay

    Args:
      times  (list int) - 120hz counter for data
      data   (list)     - data at each timestamp
      delays (list)     - what delays to form

    Example:
      >>> times = [1,3,4,5,8,11]
      >>> dataList = [1,3,4,5,8,11]
      >>> delays = [1,2,3,5,8,80]
      >>> makePairsAnswer(times, dataList, delays)
      {1: [(3, 4), (4, 5)],
       2: [(1, 3), (3, 5)],
       3: [(1, 4), (5, 8), (8, 11)],
       5: [(3, 8)],
       8: [(3, 11)],
       80: []}

    Returns:
      (dict):
         keys - delays from delays arg for which there is data
         values - list of tuple pairs. Each pair is a set of values from dataList
                  for which the times are delay (the key) apart
    '''
    assert len(times)==len(data)
    assert len(times)==len(set(times)), "times values are not unique"
    pairs = {}
    for delay in delays:
        pairs[delay]=[]
    times = [x for x in times]
    times.sort()
    for idxA, tmA in enumerate(times):
        for idxB in range(idxA+1, len(times)):
            tmB = times[idxB]
            delay = tmB-tmA
            if delay not in delays:
                continue
            pairs[delay].append((data[idxA],data[idxB]))
            
    for delay in pairs:        
        pairs[delay].sort()
    return pairs


class CorrelationCalcCallback(object):
    '''callback class for WorkerData

    Keeps the delay pairs up to date.

    ARGS:
      delay (list, int): the list of delays to do
      doRemove (bool): if True, responds to workerBeforeDataRemove and removes
                       times
      tester (unittest.TestCase): may call testing methods during callbacks
    '''
    def __init__(self, delays, doRemove, tester):
        self.delays = delays
        self.doRemove = doRemove
        self.tester = tester
        self.pairs = {}
        for delay in self.delays:
            self.pairs[delay]=[]

    def workerBeforeDataRemove(self, tm, xInd, wd):
        '''tm is the earliest time in the data, it is going to be overwritten
        '''
        if not self.doRemove: return
        maxTime = wd.maxTimeForStoredData()
        for delay in self.delays:
            laterTm = tm + delay
            delayToLargeForCurrentData = laterTm > maxTime
            if delayToLargeForCurrentData: break
            laterXind = wd.tm2idx(laterTm)
            if laterXind is None: continue
            pairForMode = (wd.X[xInd,0], wd.X[laterXind,0]) 
            self.tester.assertTrue(pairForMode in self.pairs[delay])
            self.pairs[delay].remove(pairForMode)

        minTime = wd.minTimeForStoredData()
        for delay in self.delays:
            earlierTm = tm - delay
            delayToLargeForCurrentData = earlierTm < maxTime
            if delayToLargeForCurrentData: break
            earlierXind = wd.tm2idx(earlierTm)
            if earlierXind is None: continue
            pairForMode = (wd.X[earlierXind,0], wd.X[xInd,0]) 
            self.tester.assertTrue(pairForMode in self.pairs[delay])
            self.pairs[delay].remove(pairForMode)

    def workerAdjustData(self, data):
        pass

    def workerAfterDataInsert(self, tm, xInd, wd):
        '''tm is for a new piece of data, xInd is where it is in wd.X
        It is probably the latest tm in WorkerData, but maybe not
        '''
        minTime = wd.minTimeForStoredData()
        for delay in self.delays:
            earlierTm = tm - delay
            delayToLargeForCurrentData = earlierTm < minTime
            if delayToLargeForCurrentData: break
            earlierXind = wd.tm2idx(earlierTm)
            if earlierXind is None: continue
            pairForMode = (wd.X[earlierXind,0], wd.X[xInd,0])
            self.pairs[delay].append(pairForMode)

        maxTime = wd.maxTimeForStoredData()
        for delay in self.delays:
            nextTm = tm + delay
            delayToLargeForCurrentData = nextTm > maxTime
            if delayToLargeForCurrentData: break
            nextXind = wd.tm2idx(nextTm)
            if nextXind is None: continue
            pairForMode = (wd.X[xInd,0], wd.X[nextXind,0])
            self.pairs[delay].append(pairForMode)

def mkDataAdd10(tm, numDataPointsThisWorker):
    arr = np.zeros(numDataPointsThisWorker)
    arr[:]=tm+10
    return arr

class WorkerDataCorrelationCalc( unittest.TestCase ) :

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

    def test_noRemoveGetAll(self):
        '''Make num times long enough to get all the delays, and check that they are all there
        '''
        longMessage = True
        logger = corAna.makeLogger(isTestMode=True,isMaster=True,isViewer=True,isServer=True,rank=0)
        isFirstWorker = True
        numTimes = 50
        numDataPointsThisWorker = 10
        storeDtype = np.float32
        delays = [1,2,3,5,8,10]
        corrCallback = CorrelationCalcCallback(delays, doRemove=False, tester=self)

        workerData = corAna.WorkerData(logger, isFirstWorker, 
                                       numTimes, 
                                       numDataPointsThisWorker, 
                                       storeDtype = storeDtype,
                                       addRemoveCallbackObject = corrCallback)
        
        times = range(37)
        datas = [mkDataAdd10(tm, numDataPointsThisWorker) for tm in times]
        datasForAnswer = [arr[0] for arr in datas]

        for tm,data in zip(times,datas):
            workerData.addData(tm,mkDataAdd10(tm, numDataPointsThisWorker))

        answer = makePairsAnswer(times, datasForAnswer, delays)
        callbackAnswer = corrCallback.pairs
        
        self.assertEqual(answer.keys(), callbackAnswer.keys())
        for key, dataPairsAnswer in answer.iteritems():
            dataPairsCallback = callbackAnswer[key]
            self.assertEqual(len(dataPairsCallback), len(dataPairsAnswer), msg="delay=%d, answer=%r callback=%r" % \
                             (key, dataPairsAnswer, dataPairsCallback))
        

    def test_noRemovePermuteGetAll(self):
        '''Mix up the order a bit, 
        '''
        longMessage = True
        logger = corAna.makeLogger(isTestMode=True,isMaster=True,isViewer=True,isServer=True,rank=0)
        isFirstWorker = True
        numTimes = 50
        numDataPointsThisWorker = 10
        storeDtype = np.float32
        delays = [1,2,3,5,8,10]
        corrCallback = CorrelationCalcCallback(delays, doRemove=False, tester=self)

        workerData = corAna.WorkerData(logger, isFirstWorker, 
                                       numTimes, 
                                       numDataPointsThisWorker, 
                                       storeDtype = storeDtype,
                                       addRemoveCallbackObject = corrCallback)
        
        times = range(37)
        datas = [mkDataAdd10(tm, numDataPointsThisWorker) for tm in times]
        datasForAnswer = [arr[0] for arr in datas]
        
        swapTimes = [tm for tm in times]
        swapDatas = [arr.copy() for arr in datas]

        swapList = [(0,1), (2,4), (9,10), (13,16)]
        for swapPair in swapList:
            a,b = swapPair
            tmp = swapTimes[a]
            swapTimes[a]=swapTimes[b]
            swapTimes[b]=tmp
            tmp = swapDatas[a]
            swapDatas[a]=swapDatas[b]
            swapDatas[b]=tmp


        for tm,data in zip(swapTimes,swapDatas):
            workerData.addData(tm,data)

        answer = makePairsAnswer(times, datasForAnswer, delays)
        callbackAnswer = corrCallback.pairs
        
        self.assertEqual(answer.keys(), callbackAnswer.keys())
        for key, dataPairsAnswer in answer.iteritems():
            dataPairsCallback = callbackAnswer[key]
            self.assertEqual(len(dataPairsCallback), len(dataPairsAnswer), msg="delay=%d, answer=%r callback=%r" % \
                             (key, dataPairsAnswer, dataPairsCallback))
        

    def test_noRemoveGetSome(self):
        '''Make num times too short to catch all delays
        '''
        longMessage = True
        logger = corAna.makeLogger(isTestMode=True,isMaster=True,isViewer=True,isServer=True,rank=0)
        isFirstWorker = True
        numTimes = 5
        numDataPointsThisWorker = 10
        storeDtype = np.float32
        delays = [1,2,3,5,8, 10]  # with numTimes == 5, won't get anything for 5,8,10
        corrCallback = CorrelationCalcCallback(delays, doRemove=False, tester=self)

        workerData = corAna.WorkerData(logger, isFirstWorker, 
                                       numTimes, 
                                       numDataPointsThisWorker, 
                                       storeDtype = storeDtype,
                                       addRemoveCallbackObject = corrCallback)
        
        def mkDataAdd10(tm, numDataPointsThisWorker):
            arr = np.zeros(numDataPointsThisWorker)
            arr[:]=tm+10
            return arr
            
        times = range(37)
        datas = [mkDataAdd10(tm, numDataPointsThisWorker) for tm in times]
        datasForAnswer = [arr[0] for arr in datas]

        for tm,data in zip(times,datas):
            workerData.addData(tm,mkDataAdd10(tm, numDataPointsThisWorker))

        answer = makePairsAnswer(times, datasForAnswer, delays)
        callbackAnswer = corrCallback.pairs
        
        self.assertEqual(answer.keys(), callbackAnswer.keys())
        for key, dataPairsAnswer in answer.iteritems():
            dataPairsCallback = callbackAnswer[key]
            if key < numTimes:
                self.assertEqual(len(dataPairsCallback), len(dataPairsAnswer), msg="delay=%d, answer=%r callback=%r" % \
                                 (key, dataPairsAnswer, dataPairsCallback))
            else:
                self.assertEqual(dataPairsCallback,[])

    def test_removeGetSome(self):
        '''We should only have the last set of delays
        '''
        longMessage = True
        logger = corAna.makeLogger(isTestMode=True,isMaster=True,isViewer=True,isServer=True,rank=0)
        isFirstWorker = True
        numTimes = 5
        numDataPointsThisWorker = 10
        storeDtype = np.float32
        delays = [1,2,3,5,8,10]  # with numTimes == 5, won't get anything for 5,8,10
        corrCallback = CorrelationCalcCallback(delays, doRemove=True, tester=self)

        workerData = corAna.WorkerData(logger, isFirstWorker, 
                                       numTimes, 
                                       numDataPointsThisWorker, 
                                       storeDtype = storeDtype,
                                       addRemoveCallbackObject = corrCallback)
        
        times = range(37)
        datas = [mkDataAdd10(tm, numDataPointsThisWorker) for tm in times]
        datasForAnswer = [arr[0] for arr in datas]

        for tm,data in zip(times,datas):
            workerData.addData(tm,mkDataAdd10(tm, numDataPointsThisWorker))

        # since we have removed and numTimes is 5, we should only have pairs with
        # (32,33,34,35,36) in it
        answer = makePairsAnswer(times[-5:], datasForAnswer[-5:], delays)
        callbackAnswer = corrCallback.pairs

        self.assertEqual(answer.keys(), callbackAnswer.keys())
        for key, dataPairsAnswer in answer.iteritems():
            dataPairsCallback = callbackAnswer[key]
            self.assertEqual(len(dataPairsCallback), len(dataPairsAnswer), msg="delay=%d, answer=%r callback=%r" % \
                             (key, dataPairsAnswer, dataPairsCallback))
        

def debug():
    '''for running tests interatively.

    Example:
      # from ipytyhon shell
      import ParCorAna.unitTestsWorkerData as ut
      %pdb
      ut.debug()

    In the above example, you will break in unitTest, do one 'up' command to get back to where 
    you were in your test
    '''
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    suite.debug()

if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], '-v'])
