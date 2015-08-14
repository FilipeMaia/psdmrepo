import os
import io
import sys
import time
import datetime
import glob
import multiprocessing

__doc__='''library support for live mode testing.
In particular, two routines,

inProgressCopyWithThrottle - this is for copying a file over, simulate the data mover

simLiveMode - launches a bunch of threads doing inProgressCopyWithThrottle
'''

def inProgressCopyWithThrottle(src, dest, start_delay, mb_per_write, inprogress_ext, 
                               max_mbs, delay_between_writes, forceOverwrite, verbose, lock=None):
    '''copies a file to an inprogress file with throttling. Renames the file when done.

    Copies a src to a dest in a sequence of blocks and sleeps between blocks. The caller can control 
    the block size, time slept between writes, and initial delay. The writing should be unbufferred, 
    meaning blocks are flushed to disk immediately. With verbose=True, a print to stdout with the time 
    and destination file is made for each block write. It is intended that the user may run several
    inProgressCopyWithThrottle processes simultaneoustly with the multiprocessing package. Hence when
    verbose is True, a valid lock parameter must be passed.

    ARGS
      src - source file
      dest - destination file
      start_delay - seconds to sleep before copy is started
      mb_per_write - megabytes to write for each block
      inprogress_ext - extension to use for inprogress file
      max_mbs - maximum megabytes to write. Value <=0 means write the whole file
      delay_between_writes - seconds to sleep between each block
      forceOverwrite - overwrite the destination file, as well as destination.inprogress file
                       (if they exist)
      verbose - debugging output about each block write
      lock=None - valid lock from multiprocessing must be passed if verbose is True
    '''
    ## helper function
    def vprint(msg):
        if verbose:
            now = datetime.datetime.fromtimestamp(time.time())
            lock.acquire()
            print "dest=%s time=%4.4d-%2.2d-%2.2d::%2.2d:%2.2d:%2.2d.%6.6d: %s" % \
                (os.path.basename(dest), now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond, msg)
            lock.release()

    ##################
    # start code
    t0 = time.time()
    assert len(inprogress_ext)>0, "inprogress_ext must be string of non-zero length"
    if verbose: 
        assert lock is not None, "must pass a multiprocessing.Lock() object for lock when verbose is True"
    destInProgress = dest + inprogress_ext
    if os.path.exists(dest) or os.path.exists(destInProgress):
        if not forceOverwrite:
            raise Exception("destination file: %s or inprogress file: %s exist. Add --force to overwrite" % (dest, destInProgress))
        else:
            if os.path.exists(dest): 
                os.unlink(dest)
            if os.path.exists(destInProgress): 
                os.unlink(destInProgress)

    assert mb_per_write > 0.0
    bytesPerRead = max(1,int((1<<20)*mb_per_write))
    readAll=True
    maxBytesToRead = -1
    if max_mbs > 0:
        readAll = False
        maxBytesToRead = int((1<<20)*max_mbs)
    vprint("reading %d bytes each time, with %.2f sec between reads. readAll=%s maxBytesToRead=%s" % (bytesPerRead, delay_between_writes, readAll, maxBytesToRead))
    srcFile = io.open(src, "rb")
    destInProgressFile = io.open(destInProgress, "wb",0)  # open unbuffered to get immediate flush of bytes written
    time.sleep(start_delay)
    vprint("finished initial delay of %.2f seconds" % start_delay)
    totalBytesWritten = 0
    hitMaxToRead = False
    while True:
        block = srcFile.read(bytesPerRead)
        if len(block) == 0: 
            # end of file
            break
        if not readAll:
            if totalBytesWritten + len(block) > maxBytesToRead:
                toRead = maxBytesToRead - totalBytesWritten
                block = block[0:toRead]
                hitMaxToRead = True
        numWritten = destInProgressFile.write(block)
        totalBytesWritten += numWritten
        assert len(block)==numWritten, "only wrote %d bytes, but read %d" % (numWritten, len(block))
        vprint("wrote %d bytes" % totalBytesWritten)
        if hitMaxToRead:
            break
        time.sleep(delay_between_writes)

    io.IOBase.close(destInProgressFile)
    io.IOBase.close(srcFile)
    os.rename(destInProgress, dest)
    vprint("Finished. wrote %d bytes in %.2f sec" % (totalBytesWritten, time.time()-t0))

def parseStreamDictOption(streams, optionString, typeArg):
    '''parses a string that specifies values for a set of streams.
    ARGS:
      streams - list of streams that need values set
      optionString - string, see examples for format, comma separatered, and intervals
                     of streams can be specified
      typeArg  - such as int or float, what to cast string values to
    RETURN:
      a dict whose keys are equal to streams, and values come from parsing optionString.
      If optionString does not specify a value for all streams, an exception is thrown
    
    EXAMPLE:
      parseStreamDictOption([1,2], "0-255:34", int)     ->  {0:34, 1:34}
      parseStreamDictOption([1,2], "0:34,1:32", int)    ->  {0:34, 1:32}
      parseStreamDictOption([1,2], "0:34", int)         ->  exception raised
      parseStreamDictOption([1,2], "0-9:34,1:32", int)  ->  {0:34, 1:32}  # options parsed left to right
      parseStreamDictOption([1,2], "1:32,0-9:34", int)  ->  {0:34, 1:34}  # options parsed left to right
    '''
    stream2val = {}
    for streamOpt in optionString.split(','):
        stream,val = streamOpt.split(':')
        if stream.find('-')>=0:
            a,b = map(int,stream.split('-'))
            optStreams=range(a,b+1)
        else:
            optStreams=[int(stream)]
        for stream in streams:
            if stream in optStreams:
                stream2val[stream]=typeArg(val)
    assert all([stream in stream2val for stream in streams]), "optionString=%s does not specify a value for all these streams: %s" % (optionString, streams)

    return stream2val

def simLiveMode(inprogress_ext, run, srcdir, destdir, start_delays, mb_per_writes, 
                max_mbs, delays_between_writes, forceOverwrite, verbose):
    assert os.path.exists(srcdir)
    runFmt = '-r%4.4d' % run
    runChunk0XtcFiles = os.path.join(srcdir, '*%s*-c00*.xtc' % runFmt)
    srcChunk0Xtcs = glob.glob(runChunk0XtcFiles)
    assert len(srcChunk0Xtcs)>0, "no xtc files matching glob/wildcard pattern: %s" % runChunk0XtcFiles
    stream2xtc = {}
    for xtc in srcChunk0Xtcs:
        basename = os.path.basename(xtc)
        streamNo = int(basename.split('-s')[1].split('-c')[0])
        assert streamNo not in stream2xtc, "There are two xtcs with stream=%d for run=%d, they are %s and %s" % \
            (steamNo, run, xtc, stream2xtc[streamNo])
        stream2xtc[streamNo] = xtc
    streams = stream2xtc.keys()
    stream2start_delay=parseStreamDictOption(streams, start_delays, float)
    stream2mb_per_write=parseStreamDictOption(streams, mb_per_writes, float)
    stream2max_mbs=parseStreamDictOption(streams, max_mbs, float)
    stream2delays_between_writes=parseStreamDictOption(streams, delays_between_writes, float)

    if not os.path.exists(destdir):
        os.makedirs(destdir)
    
    # for each chunk 0 stream, start a process that copys it with throttle
    stream2process = {}
    lock = multiprocessing.Lock()
    for stream, xtcFile in stream2xtc.iteritems():
        srcFile = xtcFile
        destFile = os.path.join(destdir, os.path.basename(srcFile))
        start_delay = stream2start_delay[stream]
        mb_per_write = stream2mb_per_write[stream]
        max_mbs = stream2max_mbs[stream]
        delays_between_writes = stream2delays_between_writes[stream]
        process = multiprocessing.Process(target=inProgressCopyWithThrottle,
                                          args=(srcFile, destFile, start_delay, mb_per_write,
                                                inprogress_ext, max_mbs, delays_between_writes,
                                                forceOverwrite, verbose, lock))
        stream2process[stream]=process
        process.start()

    for stream, process in stream2process.iteritems():
        process.join()
        
    

