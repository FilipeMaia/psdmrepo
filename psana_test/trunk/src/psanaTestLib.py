# This is a script for managing psana test data. From the release directory do
#
#        python psana_test/src/psanaTestLib.py
#
# to get the usage commands that are available.

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
import xml.etree.ElementTree as ET

usage = '''enter one of the commands:

prev   updates psana_test/data/previousDump.txt.
       For each xtc in the testdata, write the md5 sum of the xtc, the dump of the full xtc,
       and if the xtc is included in the regressionTests.txt file, the specified dump (usually a
       smaller number of events, without epics aliases in the dump).

       This file is used by the test command to detect changes.

       optional args:  
         delete=False    don't delete dump file (they will be in psana_test/data/prev_xtc_dump)
         all=True        redo dump of all tests - not just new test files. 
                         Does not append, creates psana_test/data/previousDump.txt from scratch.

links  creates new soft links in /reg/g/psdm/data_test/types that link to the xtc in ../Translator
       that name a distinct psana type, or distint epics pv (dbr & numElements).
       Does not remove existing link if it is there. 
       optional args:
         noTypes   noEpics

test  default is to do regression test - the tests listed in psana_test/data/regressionTests.txt.
      For those files, it runs psana_test.dump on the specified number of events (without dumping 
      epics aliases) and compares to the md5sum to that listed in previousDump.txt. Then it translates 
      the specified number of events to hdf5 and runs psana_test.dump again. This output is compared
      with the previous output. Some differences are expected, but other differences raise errors.
      Expected differences are recorded in regressionTestsExpectedDiffs.xml

       optional args:
         verbose=True delete=False
         set=n  or n,m,k            just to those regression test numbers
         set=full                   not the regression, processes complete xtc of all tests
         set=full:n or full:n,m,k   a select number of full tests to do

types  find xtc with new types. Make new test_0xx files. Updates psana_test/data/regressionTests.txt.
       optional args: 
         datagrams=n  number of datagrams to look at with each new file (remember, each stream is 20hz)
         regress      just generated the regression test file

curtypes report on current types
'''

##############################################################
# to make the number of pds type id's available to python, we wrap a C++ function
# that returns the value:
dll = ctypes.cdll.LoadLibrary('libpsana_test.so')
psana_test_getNumberOfPdsTypeIds = dll.getNumberOfPdsTypeIds
psana_test_getNumberOfPdsTypeIds.restype = ctypes.c_int
psana_test_getNumberOfPdsTypeIds.argtypes = []

def getNumberOfPdsTypeIds():
    return psana_test_getNumberOfPdsTypeIds()


############################################################
# Exceptions:
class BadXtcFilename(Exception):
    def __init__(self,msg=''):
        super(BadXtcFilename, self).__init__(msg)

# alarm_handler for timeout when running shell commands
class Alarm(Exception):
    pass

def alarm_handler(signum, frame):
    raise Alarm

###########################################################
# testing functions
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
    of the form test_xxx_*.xtc where xxx is an integer.

    optional arg: noTranslator=True means other than test 42, don't 
    return tests with Translator in the name.

    Returns a dict:
      res[n]={'basname':basename, 'path':path}
    where n is the testnumber, 
          basename  is the file basename, such as test_000_exp.xtc
          path      is the full path to the xtc file.
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

def getPreviousDumpFilename():
    return os.path.join('psana_test', 'data', 'previousDump.txt')

def getRegressionTestFilename():
    return os.path.join('psana_test','data','regressionTests.txt')

def getRegressionTestExpectedDifferencesFilename():
    return os.path.join('psana_test','data','regressionTestsExpectedDiffs.xml')

def getRegressionTestExpectedDifferences():
    tree = ET.parse(getRegressionTestExpectedDifferencesFilename())
    root = tree.getroot()
    expectedDiffs = {}
    for regress in root.findall('regress'):
        expectedDiffs[int(regress.attrib['number'])]=regress.find('diff').text.strip()
    return expectedDiffs
    
def getFullTestExpectedDifferencesFilename():
    return os.path.join('psana_test','data','fullTestsExpectedDiffs.xml')

def getFullTestExpectedDifferences():
    tree = ET.parse(getFullTestExpectedDifferencesFilename())
    root = tree.getroot()
    expectedDiffs = {}
    for full in root.findall('full'):
        expectedDiffs[int(full.attrib['number'])]=full.find('diff').text.strip()
    return expectedDiffs

def getPreviousXtcDirsFilename():
    return os.path.join('psana_test', 'data', 'previousXtcDirs.txt')

def filterPsanaStderr(ln):
    ''' returns True if this line is nothing to worry about from psana output.
    Otherwise False, it is an unexpected line in psana stderr.
    '''
    ignore = ('MsgLogger: Error while parsing MSGLOGCONFIG: unexpected logging level name',
              '[warning')
    for prefix in ignore:
        if ln.startswith(prefix) or len(ln.strip())==0:
            return True
    return False

def get_md5sum(fname, verbose=False):
    '''returns md5sum on a file. Calls command line utility md5sum
    IN: fname - fuilename
    OUT: text string - md5sum.
    '''
    assert os.path.exists(fname), "filename %s does not exist" % fname
    cmd = 'md5sum %s' % fname
    if verbose: print cmd
    o,e = cmdTimeOut(cmd, seconds=15*60)
    assert len(e)==0
    flds = o.split()
    assert len(flds)==2
    assert flds[1] == fname
    return flds[0]

def psanaDump(infile, outfile, events=None, dumpEpicsAliases=False, regressDump=True, verbose=False):
    '''Runs the  psana_test.dump module on the infile and saves the output
    to outfile. Returns output to stderr from the run, filtered as per the
    filterPsanaStderr function
    '''
    numEventsStr = ''
    epicAliasStr = ''
    if events is not None and events > 0:
        numEventsStr = '-n %d' % events
    if not dumpEpicsAliases:
        epicAliasStr = '-o psana_test.dump.aliases=False'
    regressDumpStr = '-o psana_test.dump.regress_dump=%r' % regressDump
    cmd = 'psana  -c "" %s -m psana_test.dump %s %s %s' % (numEventsStr, epicAliasStr, regressDumpStr, infile)
    if verbose: print cmd
    p = sb.Popen(cmd, shell=True, stdout=sb.PIPE, stderr=sb.PIPE)
    out,err = p.communicate()
    fout = file(outfile,'w')
    fout.write(out)
    fout.close()
    errLines = [ln for ln in err.split('\n') if len(ln.strip())>0]
    filteredErrLines = [ln for ln in errLines if not filterPsanaStderr(ln)]
    filteredErr =  '\n'.join(filteredErrLines)
    return cmd, filteredErr

class XtcLine(object):
    '''Initialize with a line of output from xtclinedump to obtain an object whose 
    attributes are the field names from the line (with values from the line as well).
    Always has the attribute 'contains' that is one of:
      'dg'      the line was for a datagram,
      'xtc'     an xtc line that is not an epics pv variable
      'xtc-pv'  an xtc line for an epics pv variable 
                object will include the two additional attributes dbrType and numElements
      'unknown'
    '''
    def __init__(self,ln):
        def inthex(x):
            return int(x,16)
        self.ln=ln
        if ln.startswith('dg='):
            self.contains='dg'
            types= [int,    inthex,    str,   str,   int,   int,  inthex,inthex,  inthex,   inthex,  inthex, inthex,inthex]
            flds = ['dg=',' offset=',' tp=',' sv=',' ex=',' ev=',' sec=',' nano=',' tcks=',' fid=',' ctrl=',' vec=',' env=']
        elif ln.startswith('xtc'):
            ln = ln[len('xtc'):]
            self.contains='xtc'
            ln = ln.split(' payload=')[0]
            if ln.find('dbr=')>0:
                self.contains += '-pv'
                if ln.find('pvName=')>0:
                    ln, pvName = ln.split(' pvName=')
                    setattr(self,'pvName',pvName)
                ln, pvId = ln.split(' pvId=')
                ln, numElem = ln.split(' numElem=')
                ln, dbr = ln.split(' dbr=')
                setattr(self,'pvId',int(pvId))
                setattr(self,'numElem',int(numElem))
                setattr(self,'dbr',int(dbr))
            if ln.find(' plen=')>=0:
                ln,val = ln.split(' plen=')
                setattr(self,'plen',int(val))
            types= [int,    inthex,    inthex, inthex,    str,    int,          str,      int,      int,       inthex,     int         , int,                    str]
            flds = [' d=',' offset=',' extent=',' dmg=',' src=', ' level=', ' srcnm=',' typeid=', ' ver=', ' value=',' compr=',' compr_ver=',' type_name=']
        else:
            self.contains='unknown'
            return
        assert len(types)==len(flds)
        assert len(ln.split('='))==len(types)+1
        types.reverse()
        flds.reverse()
        for fld,tp in zip(flds,types):
            try:
                ln,val = ln.split(fld)
            except ValueError:
                print "ln=%s\nfld=%s" % (ln,fld)
                raise
            attr = fld.strip().replace('=','')
            setattr(self,attr,tp(val))
        
    def isValid(self):
        '''For xtc, returns true if damage is 0, or only the user bit for ebeam.
        For dgrams, always returns True
        '''
        if self.contains == 'dg': return True
        if self.contains == 'unknown': return False
        if self.dmg == 0: return True
        if self.typeid == 15:   
            # the type is Bld.EBeam
            if ((self.dmg & 0xFFFFFF) == (1<<14)): 
                # only user damage
                return True 
        return False
            
    def __str__(self):
        return self.ln

    def __repr__(self):
        return str(self)
        
def getEpicsTypes(xtc, numDgrams=6, getUndamaged=True):
    cmd = 'xtclinedump xtc %s' % xtc
    if numDgrams is not None and isinstance(numDgrams,int):
        cmd += ' --dgrams=%d' % numDgrams
    o,e = cmdTimeOut(cmd)    
    assert len(e)==0, "error running: %s" % cmd
    dbrNumElem = set()
    for ln in o.split('\n'):
        lnObj = XtcLine(ln)
        if getUndamaged and not lnObj.isValid(): continue
        if lnObj.contains == 'xtc-pv':
            dbrNumElem.add((lnObj.dbr,lnObj.numElem))
    return dbrNumElem

def getPsanaTypes(datasource, numEvents=160):
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
    cmd = 'psana  -c "" %s -m EventKeys %s | grep type=Psana | sort | uniq' % (numStr, datasource,)
    o,e = cmdTimeOut(cmd, timeOut)    
    nonWarnings = [ ln for ln in e.split('\n') if not filterPsanaStderr(ln) ]
    if len(nonWarnings)>0:
        raise Exception("getPsanaTypes: cmd=%s\n failed, error:\n%s" % (cmd, e))
    for ln in o.split('\n'):
        if len(ln)<5:
            continue
        typeName = ln.split('type=')[1].split(', src=')[0]
        types.add(typeName)
    return types

def lastDgramForTypes(xtc, dgrams=240, getUndamaged=True):
    '''For each distinct typid/version pair, and epics dbr/num Elem pair,
    identifies the first datagram where this occurs as valid data, and the 
    offset of the next datagram after this type.

    IN: xtc file
    OUT: typesDict, epicsDict
      where 
    typeDict key:  (typeid, version)
             value: {'first_datagram': int, 'offset_next_datagram': int (or None)}
    epicsDict keys: (dbrtype,numElements)
             value: {'first_datagram': int, 'offset_next_datagram': int (or None)}

    '''
    cmd = 'xtclinedump xtc %s --dgrams=%s' % (xtc, dgrams)
    o,e = cmdTimeOut(cmd)
    assert len(e)==0, "error:\ncmd=%s\nstderr=%s" % (cmd,e)
    earliestTypes = {}
    earliestEpics = {}
    lns = o.split('\n')
    dg2offset={}
    fileLen = os.stat(xtc).st_size
    for ln in lns:
        lnobj=XtcLine(ln)
        if lnobj.contains == 'dg':
            dg = lnobj.dg
            dg2offset[dg]=lnobj.offset
        if getUndamaged and not lnobj.isValid(): continue
        if lnobj.contains == 'xtc':
            typeVer=(lnobj.typeid, lnobj.ver)
            if typeVer not in earliestTypes:
                earliestTypes[typeVer]={'first_datagram':dg,'offset_next_datagram':None}
        elif lnobj.contains == 'xtc-pv':
            dbrNumElem = (lnobj.dbr, lnobj.numElem)
            if dbrNumElem not in earliestEpics:
                earliestEpics[dbrNumElem]={'first_datagram':dg,'offset_next_datagram':None}
    for earlyDgDict in [earliestTypes, earliestEpics]:
        for key,value in earlyDgDict.iteritems():
            dg = value['first_datagram']
            value['offset_next_datagram'] = dg2offset.get(dg+1,fileLen)
    return earliestTypes, earliestEpics

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
    assert os.path.exists(prevDir), ("Directory %s does not exist. " + \
                                     "Run from a release directory with psana_test checked out") % prevDir
    prevFullName = getPreviousDumpFilename()
    if doall:
        fout = file(prevFullName,'w')
        prev = set()
    else:
        assert os.path.exists(prevFullName), "file %s does not exist, run with all=True" % prevFullName
        prev = set(readPrevious().keys())
        fout = file(prevFullName,'a')
    testTimes = {}
    regressTests = readRegressionTestFile()
    print "** prev: carrying out md5 sum of xtc, psana_test.dump of xtc and regression tests."
    print "   (dump a few events for some xtc, without epics aliases in dump, and no epics pvId's in dump)"
    for testNumber, fileInfo in getTestFiles().iteritems():
        if testNumber in prev:
            continue
        t0 = time.time()
        baseName, fullPath = (fileInfo['basename'], fileInfo['path'])
        xtc_md5 = get_md5sum(fullPath)
        dumpBase = baseName + '.dump'
        dumpPath = os.path.join('psana_test', 'data', 'prev_xtc_dump', dumpBase)        
        cmd, err = psanaDump(fullPath, dumpPath, dumpEpicsAliases=True, regressDump=False)
        if len(err)>0:
            fout.close()
            errMsg = '** FAILURE ** createPrevious: psana_test.dump produced errors on test %d\n' % testNumber
            errMsg += 'cmd: %s\n' % cmd
            errMsg += err
            raise Exception(errMsg)
        dump_md5 = get_md5sum(dumpPath)
        if deleteDump: os.unlink(dumpPath)
        regress_md5 = '0' * 32
        if testNumber in regressTests:
            testInfo = regressTests[testNumber]
            regressXtc, events, dumpEpicsAliases = testInfo['path'], testInfo['events'], \
                                                   testInfo['dumpEpicsAliases']            
            assert regressXtc == fullPath, "regression test %d xtc != testData xtc, regress=%s testData=%s" % \
                (testNumber, regressXtc, fullPath)
            regressDumpBase = 'regress_' + dumpBase
            regressDumpPath = os.path.join('psana_test', 'data', 'prev_xtc_dump', regressDumpBase)
            cmd, err = psanaDump(fullPath, regressDumpPath, events, dumpEpicsAliases=False, regressDump=True)
            if len(err)>0:
                fout.close()
                errMsg = '** FAILURE ** createPrevious: psana_test.dump produced errors on regress test %d\n' % testNumber
                errMsg += 'cmd: %s\n' % cmd
                errMsg += err
                raise Exception(errMsg)
            regress_md5 = get_md5sum(regressDumpPath)
            if deleteDump: os.unlink(regressDumpPath)
        fout.write("xtc_md5sum=%s   dump_md5sum=%s   regress_md5sum=%s   xtc=%s\n" % (xtc_md5, dump_md5, regress_md5, baseName))
        fout.flush()
        testTimes[testNumber] = time.time()-t0
        print "** prev:  test=%3d time=%.2f seconds" % (testNumber, testTimes[testNumber])
    fout.close()
    testsTime = sum(testTimes.values())
    print "* tests time: %.2f sec, or %.2f min" % (testsTime,testsTime/60.0)

def readPrevious():
    '''A line of this file has
    xtc_md5sum=... dump_md5sum=... regress_md5sum=... xtc=...
    which are the md5 sum of the whole xtc file, the md5 sum of a dump of the whole xtc
    then an md5 sum of a dump of the regression test for this file, or 0 if this file is not
    included in the reguression test, finally the xtc file for the test.    
    '''
    prevFilename = getPreviousDumpFilename()
    assert os.path.exists(prevFilename), "file %s doesn't exist, run prev command" % prevFilename
    res = {}
    for ln in file(prevFilename).read().split('\n'):
        if not ln.startswith('xtc_md5sum='):
            continue
        ln,xtc = ln.split(' xtc=')
        ln,md5regress = ln.split(' regress_md5sum=')
        ln,md5dump = ln.split('dump_md5sum=')
        ln,md5xtc = ln.split('xtc_md5sum=')
        xtc = xtc.strip()
        assert xtc.startswith('test_'), "xtc file doesn't start with test_: %s" % xtc
        number = int(xtc.split('_')[1])
        md5regress = md5regress.strip()
        md5dump = md5dump.strip()
        md5xtc = md5xtc.strip()
        res[number]={'xtc':xtc, 'md5xtc':md5xtc, 'md5dump':md5dump, 'md5regress':md5regress}
    return res

def makeTypeLinks(args):
    doTypes = True
    doEpics = True
    if 'noTypes' in args:
        doTypes = False
    if 'noEpics' in args:
        doEpics = False
    linksMade = set()
    for testNum, testInfo in getTestFiles(noTranslator=True).iteritems():
        basename, fullPath = (testInfo['basename'], testInfo['path'])
        if doTypes:
            types = getPsanaTypes(fullPath, None)
            for tp in types:
                if tp in linksMade: continue
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
        if doEpics:
            epicsDbrNumElem = getEpicsTypes(fullPath)
            for dbr,numElem in epicsDbrNumElem:
                epicsLnk = 'epicsPv_dbr_%d_numElem_%d' % (dbr,numElem)
                if epicsLnk in linksMade: continue
                lnk = '/reg/g/psdm/data_test/types/%s.xtc' % epicsLnk
                if os.path.exists(lnk):
                    print "    already exists, skipping %s" % lnk
                    continue
                lnkCmd = 'ln -s %s %s' % (fullPath, lnk)
                o,e = cmdTimeOut(lnkCmd)
                assert len(e)==0, "**Failure with cmd=%s\nerr=%s" % (lnkCmd,e)
                print lnkCmd
                linksMade.add(epicsLnk)

def getEpicsTestNumbers():
    testNumbers = set()
    for fname in glob.glob('/reg/g/psdm/data_test/types/epicsPv*.xtc'):
        base = os.path.basename(fname)
        numElem = int(base.split('_numElem_')[1].split('.xtc')[0])
        if numElem > 1:
            continue
        try:
            xtcpath = os.readlink(fname)
        except OSError:
            sys.stderr.write("Unexpected: os.readlink failed for fname=%s (not a soft link?)\n" % fname)
            continue
        xtc = os.path.basename(xtcpath)
        testNumbers.add(int(xtc.split('test_')[1].split('_')[0]))
    return list(testNumbers)

def testCommand(args):
    def checkForSameXtcFile(baseName, prevBasename, num, verbose):
        assert baseName == prevBasename, ("Test no=%d xtc mismatch: previous_dump.txt says " + \
                                          "xtc files is:\n %s\n but for files on disk, " + \
                                          "it is:\n %s") % (num, prevBasename, baseName)
        if verbose: 
            print "test %d: xtc filename is the same as what was recorded in previous_dump.txt" % num
            print "   old: %s" % prevBasename
            print "   new: %s" % baseName

    def checkForSameMd5(fullPath, md5, num, verbose, msg):
        current_md5 = get_md5sum(fullPath)
        assert current_md5 == md5, ("%s\n Test no=%d md5 do not agree.\n prev=%s\n curr=%s") % \
            (msg, num, md5, current_md5)

    def translate(infile, outfile, numEvents, testNum, verbose):
        numEventsStr = ''
        if numEvents is not None and numEvents > 0:
            numEventsStr = ' -n %d' % numEvents
        cmd = 'psana %s -m Translator.H5Output -o Translator.H5Output.output_file=%s -o Translator.H5Output.overwrite=True %s'
        cmd %= (numEventsStr, outfile, infile)
        if verbose: print cmd
        o,e = cmdTimeOut(cmd,20*60)
        e = '\n'.join([ ln for ln in e.split('\n') if not filterPsanaStderr(ln)])
        if len(e) > 0:
            errMsg =  "**Failure** test %d: Translator failure" % testNum
            errMsg += "cmd=%s\n" % cmd
            errMsg += "\n%s" % e
            raise Exception(errMsg)
        if verbose: print "test %d: translation finished, produced: %s" % (num, outfile)

    def compareXtcH5Dump(currentXtcDumpPath, h5DumpPath, num, verbose, expectedOutput=''):
        cmd = 'diff %s %s' % (currentXtcDumpPath, h5DumpPath)
        if verbose: print cmd
        o,e = cmdTimeOut(cmd,5*60)
        assert len(e)==0, "** FAILURE running cmd: %s\nstder:\n%s" % (cmd,e)
        if o.strip() != expectedOutput:
            msg = "** FAILURE: diff in xtc dump and h5 dump test=%d not equal to expected output\n" % num
            msg += " cmd= %s\n" % cmd
            msg += "-- diff output: --\n"
            msg += o
            if len(expectedOutput)>0:
                msg += "-- expected output: --\n"
                msg += expectedOutput
            raise Exception(msg)        
        if verbose: print "test %d: compared dump of xtc and dump of h5 file" % num

    def removeFiles(files):
        for fname in files:
            assert os.path.exists(fname)
            os.unlink(fname)

    # end helper functions
    delete,verbose,testSet = parseArgs(args,['delete','verbose', 'set'])
    assert delete.lower() in ['','false','true']
    assert verbose.lower() in ['','false','true']        

    delete = not (delete.lower() == 'false')
    verbose = (verbose.lower() == 'true')
    testFiles = getTestFiles(noTranslator=True)
    prev = readPrevious()
    regress = readRegressionTestFile()
    whichTest = 'regress'
    testNumberFilter = regress.keys()
    expectedDiffs = getRegressionTestExpectedDifferences()
    if testSet.startswith('full'):
        whichTest = 'full'
        expectedDiffs = getFullTestExpectedDifferences()
        jnk, afterFull = testSet.split('full')
        if len(afterFull)>0:
            assert afterFull.startswith(':'), "must follow full with : to specify tests, not '%s' " % afterFull
            jnk, testSet = afterFull.split(':')
    if testSet != '':
        commaSepTestSet = testSet.split(',')
        testNumberFilter = []
        for val in commaSepTestSet:
            if val.find('-')>0:
                a,b = map(int,val.split('-'))
                testNumberFilter.extend(range(a,b+1))
            else:
                testNumberFilter.append(int(val))

    for num in testNumberFilter:
        if num in testFiles:
            if (whichTest == 'full') or (whichTest == 'regress' and num in regress):
                assert num in prev, "There is a new xtc test number: %s\n. Run prev command first." % num
    testTimes = {}
    for num in testNumberFilter:
        if num not in testFiles: continue
        fileInfo = testFiles[num]
        baseName, fullPath = (fileInfo['basename'], fileInfo['path'])
        prv = prev[num]
        prevBasename, md5xtc, md5dump, md5regress = (prv['xtc'], prv['md5xtc'], 
                                                     prv['md5dump'], prv['md5regress'])
        t0 = time.time()
        checkForSameXtcFile(baseName, prevBasename, num, verbose)
        checkForSameMd5(fullPath, md5xtc, num, verbose,
                        "**DATA INTEGRITY - previously recorded md5 and current for SAME xtc file")
        if verbose: print "test %d: previously recorded md5 for xtcfile same with new md5" % num
        regressStr = ''
        numEvents = 0
        dumpEpicsAliases = True
        prevMd5 = md5dump
        if whichTest == 'regress': 
            if num not in regress:
                print "warning: test number %d not in regress tests, skipping" % num
                continue
            regressStr = '.regress'
            numEvents = regress[num]['events']
            dumpEpicsAliases = False
            prevMd5 = md5regress
        currentXtcDumpPath = os.path.join('psana_test', 'data', 'current_xtc_dump', baseName + regressStr + '.dump')
        cmd, err = psanaDump(fullPath, currentXtcDumpPath, numEvents, dumpEpicsAliases=False, regressDump=True, verbose=verbose)
        if len(err) > 0:
            raise Exception("**Failure: test=%d, psanaDump failed on xtc.\n cmd=%s\n%s" % (num, cmd, err))
        checkForSameMd5(currentXtcDumpPath, prevMd5, num, verbose, 
                        ("**FAIL - md5 of dump of xtc does not agree with prev %s" % regressStr))
        if verbose: 
            print "test %s %d: previously recorded md5 of dump of xtcfile same as new md5 of dump" % \
                (regressStr, num)
        h5dir = os.path.join('psana_test', 'data', 'current_h5')
        assert os.path.exists(h5dir), "h5dir: %s doesn't exist" % h5dir
        h5baseName = os.path.splitext(baseName)[0] + regressStr + '.h5'
        h5file = os.path.join(h5dir, h5baseName)
        translate(fullPath, h5file, numEvents, num, verbose)
        h5dumpBasename =  h5baseName + '.dump'
        h5DumpPath = os.path.join('psana_test', 'data', 'current_h5_dump', h5dumpBasename)
        cmd, err = psanaDump(h5file, h5DumpPath, numEvents, dumpEpicsAliases, regressDump=True, verbose=True)
        if len(err) > 0:
            raise Exception("**Failure: test=%d, psanaDump failed on h5.\n cmd=%s\n%s" % (num, cmd, err))
        xtc2h5ExpectedDiff = expectedDiffs.get(num,'')
        compareXtcH5Dump(currentXtcDumpPath, h5DumpPath, num, verbose, expectedOutput=xtc2h5ExpectedDiff)
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

def getValidTypeVersions(fullPath, dgrams=-1, getCompressedAndUncompressedVersions=False):
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
        if not getCompressedAndUncompressedVersions:
            version = 0x7FFF & version
        typeVersions.add((typeid,version))
    return typeVersions

def getValidTypeVerFromXtcLineDumpLine(origLn, xtcFileName=''):
    '''Takes line of output from xtclinedump and returns 
    (typeId, version) for valid data. Data is valid if  payload printed and 
    typeid is < number of pds type ids. 
    
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
    ln, version = ln.split(' ver=')
    ln, typeid = ln.split(' typeid=')
    typeid, version = map(int, (typeid, version))
    if typeid > numTypes: return None
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
    '''Determines number of datagrams to copy from an xtc file to capture all the
    types listed. Copies on L1Accept beyond last datagram with new data.
    '''
    dgrams = dgrams + 7
    cmd = 'xtclinedump xtc %s --payload=1 --dgrams=%d' % (xtc, dgrams)
    o,e = cmdTimeOut(cmd, 6*60)
    assert len(e)==0, "Failure running cmd=%s\nError=%s" % (cmd,e)
    # for each type/version, need to find beginning of next datagram that is an event
    # make dictionary indexed by type/version that will go through the stages
    # 'xtc_not_seen', 
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
            tp,ver = map(int,typeVersion.split(' ver='))
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
    if isinstance(n,int):
        intervals = [[0,n]]
        print "copying %d bytes from src=%s to dest=%s" % (n,src,dest)
    elif isinstance(n,list):
        intervals = n
        print "copying %d sets of bytes:" % len(intervals),
        for interval in intervals:
            a,b = interval
            print " [%d,%d)" % (a,b),
        print " from src=%s to dest=%s" % (src, dest)
    inFile = io.open(src,'rb')
    outFile = io.open(dest,'wb')
    for interval in intervals:
        assert len(interval)==2
        a = interval[0]
        b = interval[1]
        assert isinstance(a,int)
        assert isinstance(b,int)
        inFile.seek(a)
        n = b-a
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
    print "*** types Command not fully implemented, do not run ***"
    return
    dgrams = 40
    assert len(args) in [0,1,2], "must be 0, 1 or 2 args"
    if 'regress' in args:
        makeRegressionTestFile()
        return
    if len(args)==1:
        arg = args[0]
        assert arg.startswith('dgrams='), "type optional argument is dgrams=n, not %s" % arg
        jnk,dgrams=arg.split('dgrams=')
        dgrams=int(dgrams)

    previousXtcDirsFileName = getPreviousXtcDirsFilename()
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

def makeRegressionTestFile(regressionTestFile = getRegressionTestFilename()):
    typeVerDone = set()
    epicsDone = set()
    testFiles = getTestFiles(noTranslator=True)
    eventsForTest = dict([(num,0) for num in testFiles.keys()])
    for num, basePath in testFiles.iteritems():
        base, path = basePath['basename'],basePath['path']
        fileSize = os.stat(path).st_size
        earliestTypes, earliestEpics = lastDgramForTypes(path)
        for earlyDgDict, doneSet in zip([earliestTypes, earliestEpics],
                                        [typeVerDone, epicsDone]):
            for typeId,dgDict in earlyDgDict.iteritems():
                if typeId not in doneSet:
                    dg = dgDict['first_datagram']
                    eventsForTest[num] = max(max(6,dg+1), eventsForTest[num])
                    doneSet.add(typeId)
    if os.path.exists(regressionTestFile):
        sys.stderr.write("WARNING: overwriting: %s\n" % regressionTestFile)
    fout = file(regressionTestFile,'w')
    testNumbers = testFiles.keys()
    testNumbers.sort()
    for num in testNumbers:
        basePath = testFiles[num]['path']
        events = eventsForTest[num]
        if events > 0:
            fout.write("test=%3.3d  events=%3.3d doNotDumpEpicsAliases=%d xtc=%s\n" % \
                       (num, events, 1, basePath))
    fout.close()

def readRegressionTestFile(regressionTestFile = getRegressionTestFilename()):
    assert os.path.exists(getRegressionTestFilename()), "regression file %s does not exist. Run types regress" % getRegressionTestFilename()
    fin = file(regressionTestFile)
    regressTests = {}
    for ln in fin:
        ln = ln.strip()
        if len(ln)==0: continue
        ln,xtc = ln.split(' xtc=')
        ln,noEpicsAliases=ln.split(' doNotDumpEpicsAliases=')
        ln,events=ln.split(' events=')
        ln,testno = ln.split('test=')
        events = int(events)
        noEpicsAliases = int(noEpicsAliases)
        testno = int(testno)
        regressTests[testno]={'events':events,'path':xtc,'dumpEpicsAliases':not bool(noEpicsAliases)}
    return regressTests

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
