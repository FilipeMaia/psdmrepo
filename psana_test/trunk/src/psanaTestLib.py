import sys
import glob
import subprocess as sb
import os
import logging
import datetime
import random
import io

usage = '''enter one of the commands:

prev   updates psana_test/data/previous_dump.txt.
       File contains md5sum of xtc, and of the dump of the xtc. File is used by test to detect changes.
       optional args:  prev delete=False all=True    
       where delete=False means don't delete dump file and all=True means don't append to previous_dump.txt,
       create from scratch.

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
    cmd = 'md5sum %s' % fname
    p = sb.Popen(cmd, shell=True, stdout=sb.PIPE, stderr=sb.PIPE)
    o,e = p.communicate()
    assert len(e)==0
    flds = o.split()
    assert len(flds)==2
    assert flds[1] == fname
    return flds[0]

def psanaDump(infile, outfile):
    '''Runs the  psana_test.dump module on the infile and saves the output
    to outfile. Returns output to stderr from the run, filtered as per the
    errFilterStartsWith option. 
    '''
    cmd = 'psana -m psana_test.dump %s' % infile
    p = sb.Popen(cmd, shell=True, stdout=sb.PIPE, stderr=sb.PIPE)
    out,err = p.communicate()
    fout = file(outfile,'w')
    fout.write(out)
    fout.close()
    errLines = err.split('\n')
    filteredErrLines = [ln for ln in errLines if not filterPsanaStderr(ln)]
    filteredErr =  '\n'.join(filteredErrLines)
    return cmd, filteredErr

def getPsanaTypes(datasource, numEvents=120):
    '''Returns a set of the Psana types in the first numEvents of the datasource
    '''
    types = set()
    if numEvents == None:
        numStr=''
    else:
        numStr='-n %d' % numEvents
    cmd = 'psana %s -m EventKeys %s | grep type=Psana | sort | uniq' % (numStr, datasource,)
    p = sb.Popen(cmd, shell=True, stdout=sb.PIPE, stderr=sb.PIPE)
    o,e = p.communicate()
    nonWarnings = [ ln for ln in e.split('\n') if not filterPsanaStderr(ln) ]
    if len(nonWarnings)>0:
        raise Exception("getPsanaTypes: cmd=%s\n failed, error:\n%s" % (cmd, e))
    for ln in o.split('\n'):
        if len(ln)<5:
            continue
        typeName = ln.split('type=')[1].split(', src=')[0]
        types.add(typeName)
    return types
    
def rangeStrToList(rangeStr):
    '''Takes a string like 3-5,8 and returns [3,4,5,8]
    '''
    rangeVals = []
    flds = rangeStr.split(',')
    for fld in flds:
        if fld.find('-')>0:
            a,b = map(int,fld.split('-'))
            rangeVals.extend(range(a,b+1))
        else:
            rangeVals.append(int(fld))
    rangeVals = list(set(rangeVals))
    rangeVals.sort()
    return rangeVals

def parseArgs(args,cmds):
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
    for testNumber, fileInfo in getTestFiles().iteritems():
        if testNumber in prev:
            continue
        print "starting previous test %d" % testNumber
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
        print "** dumped test %d" % testNumber
    fout.close()
             
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
        types = getPsanaTypes(fullPath)
        for tp in types:
            if tp in linksMade:
                continue
            tpForFileName = tp.replace('Psana::','')
            tpForFileName = tpForFileName.replace('::','_')
            lnk = '/reg/g/psdm/data_test/types/%s.xtc' % tpForFileName
            if os.path.exists(lnk):
                print "already exists, skipping %s" % lnk
                continue
            lnkCmd = 'ln -s %s %s' % (fullPath, lnk)
            p = sb.Popen(lnkCmd, shell=True, stdout=sb.PIPE, stderr=sb.PIPE)
            o,e = p.communicate()
            assert len(e)==0, "**Failure with cmd=%s\nerr=%s" % (lnkCmd,e)
            print lnkCmd
            linksMade.add(tp)

def testCommand(args):
    def checkForSameXtcFile(baseName, prevBasename, num):
        assert baseName == prevBasename, ("Test no=%d xtc mismatch: previous_dump.txt says " + \
                                          "xtc files is:\n %s\n but for files on disk, " + \
                                          "it is:\n %s") % (num, prevBasename, baseName)
    def checkForSameMd5(fullPath, md5, num, msg):
        current_md5 = get_md5sum(fullPath)
        assert current_md5 == md5, ("%s\n Test no=%d md5 do not agree.\n prev=%s\n curr= %s") % \
            (msg, num, md5, current_md5)

    def checkForSameXtcDumpMd5(baseName, fullPath, num):
        dumpBase = baseName + '.dump'
        dumpPath = os.path.join('psana_test', 'data', 'current_xtc_dump', dumpBase)        
        cmd, err = psanaDump(fullPath, dumpPath)
        if len(err)>0:
            errMsg = '** FAILURE ** test - psana_Test.dump produced errors on test %d\n' % num
            errMsg += 'cmd: %s\n' % cmd
            errMsg += err
            raise Exception(errMsg)
        m5curr = get_md5sum(dumpPath)
        
        return dumpPath
            
    def translate(infile, outfile, num):
        cmd = 'psana -m Translator.H5Output -o Translator.H5Output.output_file=%s -o Translator.H5Output.overwrite=True %s'
        cmd %= (outfile, infile)
        p = sb.Popen(cmd, shell=True, stdout=sb.PIPE, stderr=sb.PIPE)
        o,e = p.communicate()
        e = '\n'.join([ ln for ln in e.split('\n') if not filterPsanaStderr(ln)])
        if len(e) > 0:
            errMsg =  "**Failure** test %d: Translator failure" % num
            errMsg += "cmd=%s\n" % cmd
            errMsg += "\n%s" % e
            raise Exception(errMsg)

    def compareXtcH5Dump(currentXtcDumpPath, h5DumpPath, num):
        cmd = 'diff -u %s %s' % (currentXtcDumpPath, h5DumpPath)
        p = sb.Popen(cmd, shell=True, stdout=sb.PIPE, stderr=sb.PIPE)
        o,e = p.communicate()
        if len(e)>0:
            msg = "** FAILURE - xtc vs. h5 dump. test=%d" % num
            msg += "cmd= %s\n" % cmd
            msg += "-- output: --\n"
            msg += e
            raise Exception(msg)

    def removeFiles(files):
        for fname in files:
            assert os.path.exists(fname)
            os.unlink(fname)

    # end helper functions
    delete, = parseArgs(args,['delete'])
    assert delete.lower() in ['','false','true']
    if delete.lower() == 'false':
        delete = False
    else:
        delete = True # default
    testFiles = getTestFiles(noTranslator=True)
    prev = readPrevious()
    for num in testFiles:
        assert num in prev, "There is a new xtc test number: %s\n. Run prev command first." % num

    for num, fileInfo in testFiles.iteritems():
        baseName, fullPath = (fileInfo['basename'], fileInfo['path'])
        prevBasename, md5xtc, md5dump = (prev[num]['xtc'], prev[num]['md5xtc'], prev[num]['md5dump'])
        print "test number %d" % num
        checkForSameXtcFile(baseName, prevBasename, num)
        checkForSameMd5(fullPath, md5xtc, num, 
                        "**DATA INTEGRITY - previously recorded md5 and current for SAME xtc file")
        currentXtcDumpPath = os.path.join('psana_test', 'data', 'current_xtc_dump', baseName + '.dump')
        cmd, err = psanaDump(fullPath, currentXtcDumpPath)
        if len(err) > 0:
            raise Exception("**Failure: test=%d, psanaDump failed on xtc.\n cmd=%s\n%s" % (num, cmd, err))
        checkForSameMd5(currentXtcDumpPath, md5dump, num, "**FAIL - md5 of dump of xtc does not agree with prev")
        h5dir = os.path.join('psana_test', 'data', 'current_h5')
        assert os.path.exists(h5dir), "h5dir: %s doesn't exist" % h5dir
        h5baseName = os.path.splitext(baseName)[0] + '.h5'
        h5file = os.path.join(h5dir, h5baseName)
        translate(fullPath, h5file, num)
        h5dumpBasename =  h5baseName + '.dump'
        h5DumpPath = os.path.join('psana_test', 'data', 'current_h5_dump', h5dumpBasename)
        cmd, err = psanaDump(h5file, h5DumpPath)
        if len(err) > 0:
            raise Exception("**Failure: test=%d, psanaDump failed on h5.\n cmd=%s\n%s" % (num, cmd, err))
        compareXtcH5Dump(currentXtcDumpPath, h5DumpPath, num)
        if delete:
            removeFiles([currentXtcDumpPath, h5DumpPath, h5file])

def curTypesCommand(args):
    curTypes = getDataTestTypeVersions()
    print "There are %d current typeid/version pairs" % len(curTypes)

def getNewTypeVersions(xtcFileName, prevTypeVerSet, numDatagrams=40):
    typeVerions = getTypeVersions(xtc, dgrams)
    newTypeVers = typeVersions.difference(currentTypeVersions)

    cmd = 'xtclinedump xtc %s' % fullPath
    p = sb.Popen(cmd, shell=True, stdout=sb.PIPE, stderr=sb.PIPE)
    o,e = p.communicate()
    assert len(e)==0, "Failure running cmd=%s\nError=%s" % (cmd,e)
    for ln in o.split('\n'):
        flds = ln.split(' value=')
        if len(flds) != 2: continue
        ln = flds[0]
        ln, version = ln.split(' version=')
        ln, typeid = ln.split(' typeid=')
        typeid, version = map(int, (typeid, version))
        typeVersions.add((typeid,version))
    return typeVersions    

def getTypeVersions(fullPath, dgrams=-1):
    typeVersions = set()
    cmd = 'xtclinedump xtc %s --payload=1' % fullPath
    if dgrams > -1:
        cmd += ' --dgrams=%d' % dgrams
    p = sb.Popen(cmd, shell=True, stdout=sb.PIPE, stderr=sb.PIPE)
    o,e = p.communicate()
    assert len(e)==0, "Failure running cmd=%s\nError=%s" % (cmd,e)
    for ln in o.split('\n'):
        flds = ln.split(' value=')
        if len(flds) != 2: continue
        ln = flds[0]
        ln, version = ln.split(' version=')
        ln, typeid = ln.split(' typeid=')
        typeid, version = map(int, (typeid, version))
        typeVersions.add((typeid,version))
    return typeVersions

def getDataTestTypeVersions():
    print "finding type/version info in test data"
    xtcDict = getTestFiles(noTranslator=True)
    testTypeVersions = set()
    for testNum, xtcInfo in xtcDict.iteritems():
        basename, fullPath = xtcInfo['basename'], xtcInfo['path']
        typeVersions = getTypeVersions(fullPath, dgrams=-1)
        testTypeVersions = testTypeVersions.union(typeVersions)
    return testTypeVersions

def testUpdateTestData():
    updateTestData('/reg/d/psdm/mob/mob30114/xtc/e459-r0145-s00-c00.xtc', 
                   set([(83, 1), (78, 1), (75, 1)]), 40)

def updateTestData(xtc, newTypeVers, dgrams):
    dgrams = dgrams + 1
    cmd = 'xtclinedump xtc %s --payload=1 --dgrams=%d' % (xtc, dgrams)
    p = sb.Popen(cmd, shell=True, stdout=sb.PIPE, stderr=sb.PIPE)
    o,e = p.communicate()
    assert len(e)==0, "Failure running cmd=%s\nError=%s" % (cmd,e)
    # for each type/version, need to find beginning of next datagram
    nextDataGram = {}
    for typeVer in newTypeVers:
        nextDataGram[typeVer]=['not_seen',None] # offset of xtc, offset of next dgram
    toFind = len(nextDataGram)
    for ln in o.split('\n'):
        if ln.startswith('dg='):
            dgstart = int(ln.split(' tp=')[0].split(' offset=')[1],16)
            
            for typeVer in nextDataGram:
                if nextDataGram[typeVer][0] == 'seen':
                    nextDataGram[typeVer][1] = dgstart
                    nextDataGram[typeVer][0] = 'finished'
                    toFind -= 1
            if toFind <= 0:
                break
                    
        elif ln.startswith('xtc'):
            typeVersion = ln.split(' value=')[0].split(' typeid=')[1]
            tp,ver = map(int,typeVersion.split(' version='))
            tpVer = (tp,ver)
            if tpVer in nextDataGram:
                if nextDataGram[tpVer][0] == 'not_seen':
                    nextDataGram[tpVer][0] = 'seen'
    assert toFind == 0, "The expected typeVersions (%r) were not found in the first %d datagrams of %s\n state=%r"   % (newTypeVers, len(dgrams), nextDataGram)
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
    bytesToCopy = max([el[1] for el in nextDataGram.values()])
    copyBytes(xtc, bytesToCopy, newTestFilePath)

def copyBytes(src, n, dest):
    print "copying %d bytes from src=%s to dest=%s" % (n,src,dest)
    inFile = io.open(src,'rb')
    outFile = io.open(dest,'wb')
    bytes = inFile.read(n)
    outFile.write(bytes)
    inFile.close()
    outFile.close()

def testUpdatePrevXtcDirs():
    previousXtcDirsFileName = os.path.join('psana_test', 'data', 'previousXtcDirs.txt')
    previousXtcDirs = readPreviousXtcDirs(previousXtcDirsFileName)
    currentXtcDirList = getCurrentXtcDirList()
    xtcDirsToScan = getXtcDirsToScan(currentXtcDirList, previousXtcDirs)
    updatePrevXtcDirs(previousXtcDirsFileName, xtcDirsToScan)

def updatePrevXtcDirs(previousXtcDirsFileName, xtcDirsToScan):
    # read comments from previous
    comments = '\n'.join([ln for ln in file(previousXtcDirsFileName).read().split('\n') if ln.strip().startswith('#')])
    previousXtcDirs = readPreviousXtcDirs(previousXtcDirsFileName)
    currentXtcDirs = previousXtcDirs
    for scannedXtcDir, scanInfo in xtcDirsToScan.iteritems():
        currentXtcDirs[scannedXtcDir] = scanInfo['mod_timestamp']
    timeStampsXtcDirs = [(v,k) for k,v in currentXtcDirs.iteritems()]
    timeStampsXtcDirs.sort()
    

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

    filesToScan = []
    for xtcDir, xtcInfo in xtcDirsToScan.iteritems():
        filesToScan.extend(xtcInfo['xtcs'])
    print "%d files to scan" % len(filesToScan)
    random.shuffle(filesToScan)
    filesScanned = 0
    for xtc in filesToScan:
        typeVersions = getTypeVersions(xtc, dgrams)
        newTypeVers = typeVersions.difference(currentTypeVersions)
        if len(newTypeVers)>0:
            print "Found %d new types (%s) in %s" % (len(newTypeVers), newTypeVers, xtc)
            updateTestData(xtc,newTypeVers,dgrams)
            currentTypeVersions = currentTypeVersions.union(newTypeVers)
        filesScanned += 1
        if filesScanned % 50 == 0:
            print "scanned %d of %d files" % (filesScanned, len(filesToScan))
    updatePrevXtcDirs(previousXtcDirsFileName, currentXtcDirList)

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

