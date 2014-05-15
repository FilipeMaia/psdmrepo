import sys
import glob
import subprocess as sb
import os
import logging
import datetime
import random
import io
import shutil
import signal
import time

import ctypes

dll = ctypes.cdll.LoadLibrary('libpsana_test.so')
psana_test_getNumberOfPdsTypeIds = dll.getNumberOfPdsTypeIds
psana_test_getNumberOfPdsTypeIds.restype = ctypes.c_int
psana_test_getNumberOfPdsTypeIds.argtypes = []

def getNumberOfPdsTypeIds():
    return psana_test_getNumberOfPdsTypeIds()

usage = '''enter one of the commands:

prev   updates psana_test/data/previous_dump.txt.
       File contains md5sum of xtc, and of the dump of the xtc. File is used by test to detect changes.
       optional args:  
         delete=False    don't delete dump file (they will be in psana_test/data/prev_xtc_dump)
         all=True        redo dump of all tests - not just new test files. 
                         Does not append, creates psana_test/data/previous_dump.txt from scratch.

links  creates new soft links in /reg/g/psdm/data_test/types to find data with a given datatype.
       Does not remove existing link if it is there.

test   read previous_dump.txt. Dump all xtc, translate, and dump again. Report unexpected differences.

types  find xtc with new types. Make new test_0xx files. 
       optional args: datagrams=n  number of datagrams to look at with each new file (remember, each stream
       is 20hz)

curtypes report on current types
'''

class BadXtcFilename(Exception):
    def __init__(self,msg=''):
        super(BadXtcFilename, self).__init__(msg)

class Alarm(Exception):
    pass

def alarm_handler(signum, frame):
    raise Alarm

def cmdTimeOut(cmd,seconds=5*60):
    '''Runs cmd with seconds timeout - raising exceptoin Alarm if timeout is hit.
    warning: runs cmd with shell=True, considered a security hazard.
    Returns stdout, stderr for command. stderr has had blank links removed.
    '''
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(seconds)  
    try:
        p = sb.Popen(cmd, shell=True, stdout=sb.PIPE, stderr=sb.PIPE)
        o,e = p.communicate()        
        signal.alarm(0)  # reset the alarm
    except Alarm:
        raise Exception("cmd: %s\n took more than %d seconds" % (cmd, seconds))
    e = [ln for ln in e.split('\n') if len(ln.strip())>0]
    return o,'\n'.join(e)
    
def getTestFiles(noTranslator=False):
    '''returns the test files as a dictionary. These are the files
    of the form test_*.xtc. Returns a dict:
      res[n]={'basname':basename, 'path':path}
    where n is the testnumber, path the full path to the xtc file.
    '''
    files = glob.glob('/reg/g/psdm/data_test/Translator/test_*.xtc')
    basenames = [os.path.basename(fname) for fname in files]
    numbers = [int(basename.split('_')[1]) for basename in basenames]
    res = {}
    for num, basename, fullpath in zip(numbers, basenames, files):
        if noTranslator:
            if num != 42 and basename.find('Translator')>=0:
                continue
        res[num] = {'basename':basename, 'path':fullpath}
    return res

def filterPsanaStderr(ln):
    ''' returns True if this line is nothing to worry about from psana output.
    Otherwise False, it is an unexpected line in psana stderr.
    '''
    ignore = ('MsgLogger: Error while parsing MSGLOGCONFIG: unexpected logging level name',
              '[warning')
    for prefix in ignore:
        if ln.startswith(prefix):
            return True
    return False

def get_md5sum(fname):
    '''returns md5sum on a file. Calls command line utility md5sum
    IN: fname - fuilename
    OUT: text string - md5sum.
    '''
    assert os.path.exists(fname), "filename %s does not exist" % fname
    o,e = cmdTimeOut('md5sum %s' % fname, seconds=15*60)
    assert len(e)==0
    flds = o.split()
    assert len(flds)==2
    assert flds[1] == fname
    return flds[0]

def psanaDump(infile, outfile):
    '''Runs the  psana_test.dump module on the infile and saves the output
    to outfile. Returns output to stderr from the run, filtered as per the
    filterPsanaStderr function
    '''
    cmd = 'psana -m psana_test.dump %s' % infile
    p = sb.Popen(cmd, shell=True, stdout=sb.PIPE, stderr=sb.PIPE)
    out,err = p.communicate()
    fout = file(outfile,'w')
    fout.write(out)
    fout.close()
    errLines = [ln for ln in err.split('\n') if len(ln.strip())>0]
    filteredErrLines = [ln for ln in errLines if not filterPsanaStderr(ln)]
    filteredErr =  '\n'.join(filteredErrLines)
    return cmd, filteredErr

def getPsanaTypes(datasource, numEvents=120):
    '''Returns a set of the Psana types in the first numEvents of the datasource.
    Runs EventKeys.
    '''
    types = set()
    if numEvents == None:
        numStr=''
        timeOut = 0
    else:
        numStr='-n %d' % numEvents
        timeOut = numEvents
    cmd = 'psana %s -m EventKeys %s | grep type=Psana | sort | uniq' % (numStr, datasource,)
    o,e = cmdTimeOut(cmd, timeOut)    
    nonWarnings = [ ln for ln in e.split('\n') if not filterPsanaStderr(ln.strip()) ]
    if len(nonWarnings)>0:
        raise Exception("getPsanaTypes: cmd=%s\n failed, error:\n%s" % (cmd, e))
    for ln in o.split('\n'):
        if len(ln)<5:
            continue
        typeName = ln.split('type=')[1].split(', src=')[0]
        types.add(typeName)
    return types
    
def parseArgs(args,cmds):
    '''INPUT:
         args  - a list of aruments, each a string,
         cmds  - a list of commands
       RETURN:
         list of cmd values, with '' if command not present
       EXAMPLE:
          both
           parseArgs(["n=5", "type=xtc"], ['n','type'])  and
           parseArgs(["type=xtc", "n=5"], ['n','type'])  return
           ('5', 'xtc')
         while
           parseArgs(["type=xtc"], ['n','type'])  returns
           ('','xtc')
         Errors are thrown if the arguments do not start out with cmd= for a cmd in cmds                    
    '''
    res = []
    assert all([arg.find('=')>=0 for arg in args]), "args format is cmd=xx ..."
    cmdsGiven = [arg.split('=')[0] for arg in args]
    for cmd in cmdsGiven:
        assert cmd in cmds, "unknown cmd=%s, allowable cmds=%s" % (cmd, cmds)
    for cmd in cmds:
        found = False
        cmdEq = '%s=' % cmd
        for arg in args:
            if arg.startswith(cmdEq):
                if not found:
                    res.append(arg.split(cmdEq)[1])
                    found = True
                else:
                    logging.warning("parseArgs: duplicate arguments for %s, latter ignored" % cmd)
        if not found:
            res.append('')
    return res
        
def previousCommand(args):
    delete, doall = parseArgs(args,['delete','all'])
    assert delete.lower() in ['','true','false'], "delete= arg must be true or false"
    assert doall.lower() in ['','true','false'], "all= arg must be true or false"
    deleteDump = True
    doAll = False
    if delete.lower() == 'false': 
        deleteDump = False
    if doall.lower() == 'true':  
        doall = True
    previousDumpFile(deleteDump, doall)

def previousDumpFile(deleteDump=True, doall=False):
    prevDir = os.path.join('psana_test','data')
    prevName = 'previous_dump.txt'
    assert os.path.exists(prevDir), ("Directory %s does not exist. " + \
                                     "Run from a release directory with psana_test checked out") % prevDir
    prevFullName = os.path.join(prevDir, prevName)
    if doall:
        fout = file(prevFullName,'w')
        prev = set()
    else:
        assert os.path.exists(prevFullName), "file %s does not exist, run with all=True" % prevFullName
        prev = set(readPrevious().keys())
        fout = file(prevFullName,'a')
    testTimes = {}
    for testNumber, fileInfo in getTestFiles().iteritems():
        if testNumber in prev:
            continue
        t0 = time.time()
        baseName, fullPath = (fileInfo['basename'], fileInfo['path'])
        xtc_md5 = get_md5sum(fullPath)
        dumpBase = baseName + '.dump'
        dumpPath = os.path.join('psana_test', 'data', 'prev_xtc_dump', dumpBase)        
        cmd, err = psanaDump(fullPath, dumpPath)
        if len(err)>0:
            fout.close()
            errMsg = '** FAILURE ** createPrevious: psana_Test.dump produced errors on test %d\n' % testNumber
            errMsg += 'cmd: %s\n' % cmd
            errMsg += err
            raise Exception(errMsg)
        dump_md5 = get_md5sum(dumpPath)
        if deleteDump: os.unlink(dumpPath)
        fout.write("xtc_md5sum=%s   dump_md5sum=%s   xtc=%s\n" % (xtc_md5, dump_md5,baseName))
        fout.flush()
        testTimes[testNumber] = time.time()-t0
        print "** prev: testfile %d, did psana_test.dump and recorded md5. time=%.2f seconds" % (testNumber, testTimes[testNumber])
    fout.close()
    testsTime = sum(testTimes.values())
    print "* tests time: %.2f sec, or %.2f min" % (testsTime,testsTime/60.0)

def readPrevious():
    prevFilename = os.path.join('psana_test', 'data', 'previous_dump.txt')
    assert os.path.exists(prevFilename), "file %s doesn't exist, run prev command" % prevFilename
    res = {}
    for ln in file(prevFilename).read().split('\n'):
        if not ln.startswith('xtc_md5sum='):
            continue
        ln,xtc = ln.split('xtc=')
        ln,md5dump = ln.split('dump_md5sum=')
        ln,md5xtc = ln.split('xtc_md5sum=')
        xtc = xtc.strip()
        assert xtc.startswith('test_'), "xtc file doesn't start with test_: %s" % xtc
        number = int(xtc.split('_')[1])
        md5dump = md5dump.strip()
        md5xtc = md5xtc.strip()
        res[number]={'xtc':xtc, 'md5xtc':md5xtc, 'md5dump':md5dump}
    return res

def makeTypeLinks(args):
    linksMade = set()
    for testNum, testInfo in getTestFiles().iteritems():
        basename, fullPath = (testInfo['basename'], testInfo['path'])
        # I believe all Translator tests are dups of 42, and can be weird, so 
        # don't make a type link to it
        if testNum != 42 and basename.find('Translator')>=0:
            continue
        types = getPsanaTypes(fullPath, None)
        for tp in types:
            if tp in linksMade:
                continue
            tpForFileName = tp.replace('Psana::','')
            tpForFileName = tpForFileName.replace('::','_')
            lnk = '/reg/g/psdm/data_test/types/%s.xtc' % tpForFileName
            if os.path.exists(lnk):
                print "    already exists, skipping %s" % lnk
                continue
            lnkCmd = 'ln -s %s %s' % (fullPath, lnk)
            o,e = cmdTimeOut(lnkCmd)
            assert len(e)==0, "**Failure with cmd=%s\nerr=%s" % (lnkCmd,e)
            print lnkCmd
            linksMade.add(tp)

def testCommand(args):
    def checkForSameXtcFile(baseName, prevBasename, num, verbose):
        assert baseName == prevBasename, ("Test no=%d xtc mismatch: previous_dump.txt says " + \
                                          "xtc files is:\n %s\n but for files on disk, " + \
                                          "it is:\n %s") % (num, prevBasename, baseName)
        if verbose: 
            print "test %d: xtc filename is the same as what was recorded in previous_dump.txt" % num
            print "   old: %s" % prevBasename
            print "   new: %s" % baseName

    def checkForSameMd5(fullPath, md5, num, msg):
        current_md5 = get_md5sum(fullPath)
        assert current_md5 == md5, ("%s\n Test no=%d md5 do not agree.\n prev=%s\n curr=%s") % \
            (msg, num, md5, current_md5)

    def translate(infile, outfile, num, verbose):
        cmd = 'psana -m Translator.H5Output -o Translator.H5Output.output_file=%s -o Translator.H5Output.overwrite=True %s'
        cmd %= (outfile, infile)
        o,e = cmdTimeOut(cmd,20*60)
        e = '\n'.join([ ln for ln in e.split('\n') if not filterPsanaStderr(ln)])
        if len(e) > 0:
            errMsg =  "**Failure** test %d: Translator failure" % num
            errMsg += "cmd=%s\n" % cmd
            errMsg += "\n%s" % e
            raise Exception(errMsg)
        if verbose: print "test %d: translation finished, produced: %s" % (num, outfile)

    def compareXtcH5Dump(currentXtcDumpPath, h5DumpPath, num, verbose):
        cmd = 'diff -u %s %s' % (currentXtcDumpPath, h5DumpPath)
        o,e = cmdTimeOut(cmd,5*60)
        if len(e)>0:
            msg = "** FAILURE - xtc vs. h5 dump. test=%d" % num
            msg += "cmd= %s\n" % cmd
            msg += "-- output: --\n"
            msg += e
            raise Exception(msg)
        if verbose: print "test %d: compared dump of xtc and dump of h5 file" % num

    def removeFiles(files):
        for fname in files:
            assert os.path.exists(fname)
            os.unlink(fname)

    # end helper functions
    delete,verbose = parseArgs(args,['delete','verbose'])
    assert delete.lower() in ['','false','true']
    assert verbose.lower() in ['','false','true']
    delete = not (delete.lower() == 'false')
    verbose = (versbose.lower() == 'true')
    testFiles = getTestFiles(noTranslator=True)
    prev = readPrevious()
    for num in testFiles:
        assert num in prev, "There is a new xtc test number: %s\n. Run prev command first." % num
    testTimes = {}
    for num, fileInfo in testFiles.iteritems():
        baseName, fullPath = (fileInfo['basename'], fileInfo['path'])
        prevBasename, md5xtc, md5dump = (prev[num]['xtc'], prev[num]['md5xtc'], prev[num]['md5dump'])
        t0 = time.time()
        checkForSameXtcFile(baseName, prevBasename, num, verbose)
        checkForSameMd5(fullPath, md5xtc, num, 
                        "**DATA INTEGRITY - previously recorded md5 and current for SAME xtc file")
        if verbose: print "test %d: previously recorded md5 for xtcfile same with new md5" % num
        currentXtcDumpPath = os.path.join('psana_test', 'data', 'current_xtc_dump', baseName + '.dump')
        cmd, err = psanaDump(fullPath, currentXtcDumpPath)
        if len(err) > 0:
            raise Exception("**Failure: test=%d, psanaDump failed on xtc.\n cmd=%s\n%s" % (num, cmd, err))
        checkForSameMd5(currentXtcDumpPath, md5dump, num, "**FAIL - md5 of dump of xtc does not agree with prev")
        if verbose: print "test %d: previously recorded md5 of dump of xtcfile same as new md5 of dump" % num
        h5dir = os.path.join('psana_test', 'data', 'current_h5')
        assert os.path.exists(h5dir), "h5dir: %s doesn't exist" % h5dir
        h5baseName = os.path.splitext(baseName)[0] + '.h5'
        h5file = os.path.join(h5dir, h5baseName)
        translate(fullPath, h5file, num, verbose)
        h5dumpBasename =  h5baseName + '.dump'
        h5DumpPath = os.path.join('psana_test', 'data', 'current_h5_dump', h5dumpBasename)
        cmd, err = psanaDump(h5file, h5DumpPath)
        if len(err) > 0:
            raise Exception("**Failure: test=%d, psanaDump failed on h5.\n cmd=%s\n%s" % (num, cmd, err))
        compareXtcH5Dump(currentXtcDumpPath, h5DumpPath, num, verbose)
        if delete:
            removeFiles([currentXtcDumpPath, h5DumpPath, h5file])
        testTime = time.time()-t0
        testTimes[num]=testTime
        print "test %3d success: total time: %.2f sec or %.2f min" % (num,testTime,testTime/60.0)

    totalTestTimes = sum(testTimes.values())
    print "total test times: %.2f sec or %.2f min" % (totalTestTimes, totalTestTimes/60.0)

def curTypesCommand(args):
    curTypes = getDataTestTypeVersions()
    print "There are %d current typeid/version pairs" % len(curTypes)

def getValidTypeVersions(fullPath, dgrams=-1):
    typeVersions = set()
    cmd = 'xtclinedump xtc %s --payload=1' % fullPath
    if dgrams > -1:
        cmd += ' --dgrams=%d' % dgrams
    o,e = cmdTimeOut(cmd,5*60)
    assert len(e)==0, "Failure running cmd=%s\nError=%s" % (cmd,e)
    for ln in o.split('\n'):
        ret = getValidTypeVerFromXtcLineDumpLine(ln, fullPath)
        if ret is None: continue
        typeid, version = ret
        typeVersions.add((typeid,version))
    return typeVersions

def getValidTypeVerFromXtcLineDumpLine(origLn, xtcFileName=''):
    '''Takes line of output from xtclinedump and returns (typeId, version) 
    if valid payload printed and typeid is < 2 * number of pds type ids. 
    
    returns (typeid, version)  on success, 
            None  otherwise
    '''
    numTypes = getNumberOfPdsTypeIds()
    flds = origLn.split(' value=')
    if len(flds) != 2: return None
    ln, afterValue = flds
    if afterValue.find('payload=') < 0: return None
    payload = afterValue.split('payload=')[1]
    if not payload.startswith('0x'): 
        dmg = int(ln.split(' src=')[0].split(' dmg=')[1], 16)
        assert dmg != 0, "unexpected, payload says damaged, but dmg is 0.\nln=%s\nxtc=%s" % \
                      (origLn, xtcFileName)
        return None
    ln, version = ln.split(' version=')
    ln, typeid = ln.split(' typeid=')
    typeid, version = map(int, (typeid, version))
    if typeid >= 2*numTypes: return None
    return (typeid, version)

def getDataTestTypeVersions():
    print "finding type/version info in test data"
    xtcDict = getTestFiles(noTranslator=True)
    testTypeVersions = set()
    for testNum, xtcInfo in xtcDict.iteritems():
        basename, fullPath = xtcInfo['basename'], xtcInfo['path']
        typeVersions = getValidTypeVersions(fullPath, dgrams=-1)
        testTypeVersions = testTypeVersions.union(typeVersions)
    return testTypeVersions

#def testUpdateTestData():
#    updateTestData('/reg/d/psdm/mob/mob30114/xtc/e459-r0145-s00-c00.xtc', 
#                   set([(83, 1), (78, 1), (75, 1)]), 40)

def updateTestData(xtc, newTypeVers, dgrams):
    dgrams = dgrams + 7
    cmd = 'xtclinedump xtc %s --payload=1 --dgrams=%d' % (xtc, dgrams)
    o,e = cmdTimeOut(cmd, 6*60)
    assert len(e)==0, "Failure running cmd=%s\nError=%s" % (cmd,e)
    # for each type/version, need to find beginning of next datagram that is an event
    l1acceptFollowing = {}
    for typeVer in newTypeVers:
        l1acceptFollowing[typeVer]=['xtc_not_seen',None] # offset of xtc, offset of next dgram
    toFind = len(l1acceptFollowing)
    for ln in o.split('\n'):
        if ln.startswith('dg='):
            lnA,lnB = ln.split(' tp=')
            sv = lnB.split(' ex=')[0].split(' sv=')[1].strip()
            if sv != 'L1Accept': continue
            dgstart = int(lnA.split(' offset=')[1],16)
            for typeVer in l1acceptFollowing:
                if l1acceptFollowing[typeVer][0] == 'xtc_seen':
                    l1acceptFollowing[typeVer][0] = 'first_l1accept'
                elif l1acceptFollowing[typeVer][0] == 'first_l1accept':
                    l1acceptFollowing[typeVer][1] = dgstart
                    l1acceptFollowing[typeVer][0] = 'finished'
                    toFind -= 1
            if toFind <= 0:
                break
        elif ln.startswith('xtc'):
            typeVersion = ln.split(' value=')[0].split(' typeid=')[1]
            payloadValid = ln.split(' payload=')[1].startswith('0x')
            tp,ver = map(int,typeVersion.split(' version='))
            tpVer = (tp,ver)
            if tpVer in l1acceptFollowing and payloadValid:
                if l1acceptFollowing[tpVer][0] == 'xtc_not_seen':
                    l1acceptFollowing[tpVer][0] = 'xtc_seen'

    assert toFind == 0, ("Two L1Accept's were not found following valid xtc " + \
                         "containing ALL the typeVersions %r within the first " + \
                         "%d datagrams of %s\n state=%r") % (newTypeVers, dgrams, l1acceptFollowing)
    xtcPath, xtcBase = os.path.split(xtc)
    expPath, xtcDir = os.path.split(xtcPath)
    assert xtcDir == 'xtc', "filenames %s does not have xtc as dir above basename" % xtc
    instPath, expDir = os.path.split(expPath)
    rootPath, inst = os.path.split(instPath)
    testFiles = getTestFiles()
    n = max(testFiles.keys())
    nextFile = n+1
    newTestFileBaseName = 'test_%3.3d_%s_%s_%s' % (nextFile, inst, expDir, xtcBase)
    newTestFilePath = os.path.join('/reg/g/psdm/data_test/Translator',newTestFileBaseName)
    bytesToCopy = max([el[1] for el in l1acceptFollowing.values()])
    copyBytes(xtc, bytesToCopy, newTestFilePath)

def copyBytes(src, n, dest):
    print "copying %d bytes from src=%s to dest=%s" % (n,src,dest)
    inFile = io.open(src,'rb')
    outFile = io.open(dest,'wb')
    bytes = inFile.read(n)
    outFile.write(bytes)
    inFile.close()
    outFile.close()

#def testUpdatePrevXtcDirs():
#    previousXtcDirsFileName = os.path.join('psana_test', 'data', 'previousXtcDirs.txt')
#    previousXtcDirs = readPreviousXtcDirs(previousXtcDirsFileName)
#    currentXtcDirList = getCurrentXtcDirList()
#    xtcDirsToScan = getXtcDirsToScan(currentXtcDirList, previousXtcDirs)
#    updatePrevXtcDirs(previousXtcDirsFileName, xtcDirsToScan)

def updatePrevXtcDirs(previousXtcDirsFileName,  xtcDirsToScan):
    # read comments from previous
    comments = '\n'.join([ln for ln in file(previousXtcDirsFileName).read().split('\n') if ln.strip().startswith('#')])
    previousXtcDirs = readPreviousXtcDirs(previousXtcDirsFileName)
    currentXtcDirs = previousXtcDirs
    for scannedXtcDir, scanInfo in xtcDirsToScan.iteritems():
        currentXtcDirs[scannedXtcDir] = scanInfo['mod_timestamp']
    timeStampsXtcDirs = [(v,k) for k,v in currentXtcDirs.iteritems()]
    timeStampsXtcDirs.sort()
    # backup previous:
    n = 1
    backup = '%s.%3.3d' % (previousXtcDirsFileName,n)
    while os.path.exists(backup):
        n += 1
        backup = '%s.%3.3d' % (previousXtcDirsFileName,n)
    shutil.copyfile(previousXtcDirsFileName, backup)
    current = file(previousXtcDirsFileName,'w')
    current.write(comments)
    current.write('\n\n')
    for timeStamp, xtcDir in timeStampsXtcDirs:
        current.write('%s %s %s\n' % (xtcDir, timeStamp, datetime.datetime.fromtimestamp(timeStamp)))
    

def readPreviousXtcDirs(fname):
    assert os.path.exists(fname), \
        "previous xtc directories scanned file: %s does not exist" % fname
    lines = [ln.strip() for ln in file(fname).read().split('\n') \
             if len(ln.strip())>0 and ln.strip()[0] != '#']
    previousXtcDirs = dict()
    for ln in lines:
        xtcDir, lastModTimeStamp, lastModDate, lastModTime = ln.split()
        previousXtcDirs[xtcDir] = float(lastModTimeStamp)
    return previousXtcDirs

def getCurrentXtcDirList():
    instrumentDirectories = ['amo','cxi','dia','mec','mob','sxr','usr','xcs','xpp']
    currentXtcDirList = []
    for instDir in instrumentDirectories:
        instPath = os.path.join('/reg/d/psdm', instDir)
        expPaths = glob.glob(os.path.join(instPath, '*'))
        for expPath in expPaths:
            xtcPath = os.path.join(expPath, 'xtc')
            if os.path.exists(xtcPath):
                currentXtcDirList.append(xtcPath)
    assert len(currentXtcDirList)>200, "Unexpected: find only found %d xtcDirs, should be over 200?" % len(currentXtcDirList)
    return currentXtcDirList

def parseXtcFileName(xtc):
    xtc = os.path.basename(xtc)
    flds = xtc.split('-')
    if len(flds) != 4:
        raise BadXtcFilename("xtc=%s and flds=%r" % (xtc, flds))
    exp,run,stream,chunk = flds
    return int(run[1:]), int(stream[1:]), int(chunk[1:].split('.xtc')[0])

def getXtcDirsToScan(currentXtcDirList, previousXtcDirs):
    print "going through %d xtc directories, checking xtc file modification times" % len(currentXtcDirList)
    xtcDirsToScan = dict()
    for xtcDir in currentXtcDirList:
        xtcs = glob.glob(os.path.join(xtcDir,'*.xtc'))
        if len(xtcs)==0: continue
        newDir = xtcDir not in previousXtcDirs
        lastModTimeStamp = previousXtcDirs.get(xtcDir,0.0)
        lastModStr = str(datetime.datetime.fromtimestamp(lastModTimeStamp))
        xtcsToScan = []
        for xtc in xtcs:
            run, stream, chunk = parseXtcFileName(xtc)
            if chunk != 0:
                continue
            xtcStat = os.stat(xtc)
            modTimeStamp = xtcStat.st_mtime
            if modTimeStamp > lastModTimeStamp:
                xtcsToScan.append((xtc,modTimeStamp))
        newFiles = False
        if len(xtcsToScan)>0:
            if not newDir: newFiles = True
            lastModTimeStamp = max([x[1] for x in xtcsToScan])
            lastModStr = str(datetime.datetime.fromtimestamp(lastModTimeStamp))
            xtcDirsToScan[xtcDir]={'xtcs':[x[0] for x in xtcsToScan],
                                   'mod_timestamp':lastModTimeStamp,
                                   'mod_str':lastModStr,
                                   'newDir':newDir,
                                   'newFiles':newFiles}
    return xtcDirsToScan

def typesCommand(args):
    dgrams = 40
    assert len(args) in [0,1], "must be 0 or 1 args"
    if len(args)==1:
        arg = args[0]
        assert arg.startswith('dgrams='), "type optional argument is dgrams=n, not %s" % arg
        jnk,dgrams=arg.split('dgrams=')
        dgrams=int(dgrams)

    previousXtcDirsFileName = os.path.join('psana_test', 'data', 'previousXtcDirs.txt')
    previousXtcDirs = readPreviousXtcDirs(previousXtcDirsFileName)
    currentXtcDirList = getCurrentXtcDirList()
    xtcDirsToScan = getXtcDirsToScan(currentXtcDirList, previousXtcDirs)
    currentTypeVersions = getDataTestTypeVersions()
    
    print "** Currently: %d typeid/versions" % len(currentTypeVersions)
    print "** %d new xtc experiment directories to scan" % len(xtcDirsToScan)
    filesToScan = []
    for xtcDir, xtcInfo in xtcDirsToScan.iteritems():
        filesToScan.extend(xtcInfo['xtcs'])
    print "%d files to scan" % len(filesToScan)
    random.shuffle(filesToScan)
    filesScanned = 0
    for xtc in filesToScan:
        typeVersions = getValidTypeVersions(xtc, dgrams)
        newTypeVers = typeVersions.difference(currentTypeVersions)
        if len(newTypeVers)>0:
            print "Found %d new types (%s) in %s" % (len(newTypeVers), newTypeVers, xtc)
            updateTestData(xtc,newTypeVers,dgrams)
            currentTypeVersions = currentTypeVersions.union(newTypeVers)
        filesScanned += 1
        if filesScanned % 50 == 0:
            print "scanned %d of %d files" % (filesScanned, len(filesToScan))
    updatePrevXtcDirs(previousXtcDirsFileName, xtcDirsToScan)

cmdDict = {
    'prev':previousCommand,
    'links':makeTypeLinks,
    'test':testCommand,
    'types':typesCommand,
    'curtypes':curTypesCommand
}
           
def main(args):
    if len(args)<2:
        print usage
        sys.exit(0)
    cmd = args[1]
    validCmds = cmdDict.keys()
    if cmd not in validCmds:
        print "ERROR: cmd '%s' is not recognized" % cmd
        print usage
        sys.exit(1)
  
    fn = cmdDict[cmd]
    fn(args[2:])

if __name__ == '__main__':
    main(sys.argv)
