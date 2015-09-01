import os
import sys
import glob
import time
import argparse
import pprint
import multiprocessing
import psana_test.liveModeSimLib as liveModeLib

programDescription = '''
Moves xtc and smalldata files from a source directory to a destination dir. 
Reads a Python config file with move parameters. 
'''

programDescriptionEpilog = '''Example Config file:

delay_xtc = 0.0
delay_smalldata = 0.0
additional_delay_xtc_stream_1 = 0.0
additional_delay_xtc_stream_5 = 0.0
additional_delay_xtc_stream_80 = 0.0
additional_delay_smalldata_stream_1 = 0.0
additional_delay_smalldata_stream_5 = 0.0
additional_delay_smalldata_stream_80 = 0.0

mb_read_xtc = 4.0
mb_read_smalldata = 0.0
additional_mb_read_xtc_stream_1 = 0.0
additional_mb_read_xtc_stream_5 = 0.0
additional_mb_read_xtc_stream_80 = 0.0
additional_mb_read_smalldata_stream_1 = 0.0
additional_mb_read_smalldata_stream_5 = 0.0
additional_mb_read_smalldata_stream_80 = 0.0

pause_between_writes_xtc = 0.0
pause_between_writes_smalldata = 0.0
additional_pause_between_writes_xtc_stream_1 = 0.0
additional_pause_between_writes_xtc_stream_5 = 0.0
additional_pause_between_writes_xtc_stream_80 = 0.0
additional_pause_between_writes_smalldata_stream_1 = 0.0
additional_pause_between_writes_smalldata_stream_5 = 0.0
additional_pause_between_writes_smalldata_stream_80 = 0.0

Note all these parameters can be overridden through command line arguments. For example:

--delay_xtc 3.3                     this introduces a 3.3 second delay between writes of the big xtc files
--delay_xtc 3.3:s0=.3:s1=-.4:s80=.4 for stream 0, delay is 3.6, stream 1, it is 2.9, and stream 80 3.7
                                    all other streams will be  3.3 seconds.  
'''

class InvalidXtcName(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)

def parseXtcFileName(xtcFile):
    baseName = os.path.basename(xtcFile)
    baseStem, ext = os.path.splitext(baseName)
    if ext != '.xtc': raise InvalidXtcName(xtcFile)
    baseStemFlds = baseStem.split('-')
    if len(baseStemFlds) < 4: raise InvalidXtcName(xtcFile)
    expStr, runStr, streamStr, chunkStr = baseStemFlds[-4:]
    if not expStr.startswith('e'): raise InvalidXtcName(xtcFile)
    if not runStr.startswith('r'): raise InvalidXtcName(xtcFile)
    if not streamStr.startswith('s'): raise InvalidXtcName(xtcFile)
    if not chunkStr.startswith('c'): raise InvalidXtcName(xtcFile)
    try:
        return (int(expStr[1:]), int(runStr[1:]), int(streamStr[1:]), int(chunkStr[1:]))
    except ValueError:
        raise InvalidXtcName(xtcFile)


def getSmallDataFile(xtcFile):
    assert xtcFile.endswith('.xtc')
    basedir, basename = os.path.split(xtcFile)
    smallDataDir = os.path.join(basedir, 'smalldata')
    if os.path.exists(smallDataDir):
        smallDataXtcBase = basename[0:-4] + '.smd.xtc'
        smallDataXtc = os.path.join(smallDataDir, smallDataXtcBase)
        if os.path.exists(smallDataXtc):
            return smallDataXtc
    return ''

def getXtcRunFiles(inputdir, run):
    xtcFiles = []
    for xtcFile in glob.glob(os.path.join(inputdir,'*.xtc')):
        try:
            parseXtcFileName(xtcFile)
            xtcFiles.append(xtcFile)
        except InvalidXtcName:
            sys.stderr.write("WARNING: bad xtcfile: %s" % xtcFile)
    runFiles = [xtcFile for xtcFile in xtcFiles if parseXtcFileName(xtcFile)[1] == run]
    if len(runFiles) == 0 and len(xtcFile) > 0:
        sys.stderr.write("ERROR: run %d doesn't exist among xtc files in %s" % (run, inputdir))
    assert len(runFiles) > 0, "no run files found for run=%d inputdir=%s" % (run, inputdir)
    return runFiles

def indexStreams(runFiles):
    stream2chunk2xtc = {}
    numberOfSmallData = 0
    for xtc in runFiles:
        exp, run, stream, chunk = parseXtcFileName(xtc)
        smallDataFile = getSmallDataFile(xtc)
        if smallDataFile != '':
            numberOfSmallData += 1
        if stream not in stream2chunk2xtc:
            stream2chunk2xtc[stream]={}
        stream2chunk2xtc[stream][chunk] = (xtc, smallDataFile)
    assert numberOfSmallData == 0 or numberOfSmallData == len(runFiles), "There are %d small data files for the %d run files" % \
        (numberOfSmallData, len(runFiles))
    stream2xtcs = {}
    for stream, chunk2xtc in stream2chunk2xtc.iteritems():
        chunks = chunk2xtc.keys()
        chunks.sort()
        xtcSmallList = [chunk2xtc[chunk] for chunk in chunks]
        stream2xtcs[stream] = {'xtc':[el[0] for el in xtcSmallList],
                               'smalldata':[el[1] for el in xtcSmallList]}
    return stream2xtcs

def getMoverParams(args, streams):
    def commandLineOverride(key, args, stream=None):
        if getattr(args,key) is None: return None
        argFields = getattr(args,key).split(':')
        try:
            overall = float(argFields[0])
            streamVals = {}
            for fld in argFields[1:]:
                streamStr, streamVal = fld.split('=')
                stream = int(streamStr[1:])
                streamVal = float(streamVal)
                streamVals[stream]=streamVal
        except:
            raise Exception("Could not parse command line argument: %s -> %s\nExample Syntax is --%s 3.3 or --%s 3.3:s0=.4:s5=-1.2:s80=.33" % (key, getattr(args,key), key, key))
        if stream is None:
            return overall
        return streamVals.get(stream,None)
            
    globalName = '__%s__' % os.path.splitext(os.path.basename(__file__))[0]
    configGlobals = { '__name__' : globalName }
    configLocals = {}
    if args.config is not None:
        execfile(args.config, configGlobals, configLocals)

    moverParams = {}
    for overallParam in ['delay', 'mb_read', 'pause_between_writes']:
        for ftype in ['xtc', 'smalldata']:
            overallKey = '%s_%s' % (overallParam, ftype)
            moverParams[overallKey] = configLocals.pop(overallKey, 0.0)
            cmdLineVal = commandLineOverride(overallKey, args)
            if cmdLineVal is not None:
                moverParams[overallKey] = cmdLineVal
            for stream in streams:
                streamKey = 'additional_%s_stream_%d' % (overallKey, stream)
                moverParams[streamKey] = configLocals.pop(streamKey, 0.0)
                cmdLineVal = commandLineOverride(overallKey, args, stream)
                if cmdLineVal is not None:
                    moverParams[streamKey] = cmdLineVal
    if moverParams['mb_read_xtc'] == 0.0:
        moverParams['mb_read_xtc'] = 4.0

    for key in configLocals.iterkeys():
        sys.stderr.write("WARNING: unknown key %s in config" % key, args.config)

    return moverParams

def getStreamMoverParams(ftype, stream, moverParams):
    params = {}
    for key in ['delay', 'mb_read', 'pause_between_writes']:
        baseKey = '%s_%s' % (key, ftype)
        offsetKey = 'additional_%s_%s_stream_%d' % (key, ftype, stream)
        params[key] = moverParams[baseKey] + moverParams[offsetKey]
    return params

def sumFileSizes(filenames):
    total = 0
    for fname in filenames:
        if not os.path.isfile(fname): return 0
        total += os.stat(fname).st_size
    return total

def computeDefaultSmallDataRead(xtcRead, xtcFiles, smallFiles):
    xtcTotalSize = sumFileSizes(xtcFiles)
    smallTotalSize = sumFileSizes(smallFiles)
    return xtcRead * (float(smallTotalSize)/float(xtcTotalSize))
    
def getAllFiles(stream2xtcFiles):
    xtcFiles = []
    smallFiles = []
    for stream, xtcSmallFiles in stream2xtcFiles.iteritems():
        xtcFiles.extend(xtcSmallFiles['xtc'])
        smallFiles.extend(xtcSmallFiles['smalldata'])
    return xtcFiles, smallFiles

def constructMoverProcess(srcFile, destFile, moveParams, timeout, verbose, lock, 
                          max_mb_write=0.0, overwrite=True):
    assert os.path.exists(srcFile), "constructMoverProcess received srcfile that doesn't exist: %s" % srcFile
    process = multiprocessing.Process(target=liveModeLib.inProgressCopyWithThrottle,
                                      args=(srcFile,
                                            destFile,
                                            moveParams['delay'],
                                            moveParams['mb_read'],
                                            '.inprogress',
                                            max_mb_write,
                                            moveParams['pause_between_writes'],
                                            overwrite,
                                            verbose,
                                            lock,
                                            timeout))
    return process

def moveAllChunks(streamFtype2chunkCurrentProcess, stream2xtcFiles, args, lock):
    outdir = {'xtc':args.outputdir, 'smalldata':os.path.join(args.outputdir, 'smalldata')}
    while True:
        time.sleep(1.0)
        stillMoving = False
        doneKeys = []
        newProcesses = {}        
        for key, startTimeProcess in streamFtype2chunkCurrentProcess.iteritems():
            stream = int(key.split('_')[-1])
            ftype = key.split('_')[0]
            t0, process = startTimeProcess
            if process.is_alive():
                stillMoving = True
            else:
                if len(stream2xtcFiles[stream][ftype])==0:
                    doneKeys.append(key)
                    continue
                srcFile = stream2xtcFiles[stream][ftype].pop(0)
                if not os.path.isfile(srcFile):
                    doneKeys.append(key)
                    continue
                destFile = os.path.join(outdir[ftype], os.path.basename(srcFile))
                moveParams = getStreamMoverParams(ftype, stream, moverParams)
                if args.timeout > 0:
                    newtimeout = max(0.0, args.timeout - (time.time()-t0))
                    if args.timeout <= 0:
                        doneKeys.append(key)
                        continue
                else:
                    newtimeout = 0.0
                newProcesses[key] = constructMoverProcess(srcFile, destFile, moveParams, newtimeout, args.verbose, lock)
        if len(newProcesses)==0 and (not stillMoving):
            break
        for key in doneKeys:
            del streamFtype2chunkCurrentProcess[key]
        for key, newProcess in newProcesses.iteritems():
            streamFtype2chunkCurrentProcess[key]=(time.time(), newProcess)
            newProcess.start()

def dataMover(args):
    runFiles = getXtcRunFiles(args.inputdir, args.run)
    stream2xtcFiles = indexStreams(runFiles)
    moverParams = getMoverParams(args, stream2xtcFiles.keys())
    if moverParams['mb_read_smalldata'] == 0.0:
        xtcFiles, smallFiles = getAllFiles(stream2xtcFiles)
        moverParams['mb_read_smalldata'] = computeDefaultSmallDataRead(moverParams['mb_read_xtc'], xtcFiles, smallFiles)
    streamFtype2chunkCurrentProcess = {}
    lock = multiprocessing.Lock()
    xtcOutDir = args.outputdir
    smalldataOutDir = os.path.join(args.outputdir, 'smalldata')
    for stream in stream2xtcFiles.keys():
        for ftype, outdir in zip(['xtc','smalldata'],[xtcOutDir, smalldataOutDir]):
            if len(stream2xtcFiles[stream][ftype])==0: continue
            srcFile = stream2xtcFiles[stream][ftype].pop(0)
            if not os.path.isfile(srcFile):
                continue
            destFile = os.path.join(outdir, os.path.basename(srcFile))
            moveParams = getStreamMoverParams(ftype, stream, moverParams)
            key = '%s_stream_%d' % (ftype, stream)
            moverProcess = constructMoverProcess(srcFile, destFile,
                                                 moveParams, args.timeout, args.verbose, lock)
            streamFtype2chunkCurrentProcess[key] = (time.time(), moverProcess)
    
    for key, startTimeProcess in streamFtype2chunkCurrentProcess.iteritems():
        t0, process = startTimeProcess
        process.start()

    try:
        moveAllChunks(streamFtype2chunkCurrentProcess, stream2xtcFiles, args, lock)
    except KeyboardInterrupt, kb:
        print "Killing current processes:"
        for timeProcess in streamFtype2chunkCurrentProcess.itervalues():
            t0, process = timeProcess
            process.terminate()
        raise kb

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=programDescription, 
                                     epilog=programDescriptionEpilog,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--inputdir', type=str, help="input directory", default=None)
    parser.add_argument('-r', '--run', type=int, help="which run to do from the input dir", default=None)
    parser.add_argument('-o', '--outputdir', type=str, help="output directory", default=None)
    parser.add_argument('-c', '--config', type=str, help="config file", default=None)
    parser.add_argument('-t', '--timeout', type=float, help="quit moving after this many seconds", default=None)
    parser.add_argument('-v', '--verbose', action='store_true', help="verbose output", default=False)

    parser.add_argument('-d', '--delay_xtc', type=str, help="xtc initial delay (override config)", default=None)
    parser.add_argument('-s', '--delay_smalldata', type=str, help="smalldata initial delay (override config)", default=None)
    parser.add_argument('-p', '--pause_between_writes_xtc', type=str, help="xtc pause between writes (override config)", default=None)
    parser.add_argument('-q', '--pause_between_writes_smalldata', type=str, help="smalldata pause between writes (override config)", default=None)
    parser.add_argument('-m', '--mb_read_xtc', type=str, help="mb xtcread (override config)", default=None)
    parser.add_argument('-n', '--mb_read_smalldata', type=str, help="mb smalldataread (override config)", default=None)
    args = parser.parse_args()

    assert args.inputdir is not None, "You must supply input directory with -i"
    assert args.outputdir is not None, "You must supply output directory with -o"
    assert args.inputdir != args.outputdir, "inputdir can't equal outputdir, this is a datamover"
    assert args.run is not None, "You must supply a run with -r"
    assert os.path.exists(args.inputdir), "Did not find input directorty: %s" % args.inputdir
    assert os.path.exists(args.outputdir), "Did not find output directorty: %s" % args.outputdir

    if args.config is not None:
        assert os.path.exists(args.config), "config file not found: %s" % args.config

    dataMover(args)
