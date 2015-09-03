# This is a script for managing and running tests with the psana test data. From the release directory do
#
#        python psana_test/src/psanaTestLib.py
#
# to get the usage commands that are available.

# Python packages
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

# ana release packages
from AppUtils.AppDataPath import AppDataPath

usage = '''enter one of the commands:

prev   updates psana_test/data/previousDump.txt.
       For each xtc in the testdata, write the md5 sum of the xtc, the dump of the full xtc,
       and if the xtc is included in the regressionTests.txt file, the specified dump (usually a
       smaller number of events, without epics aliases in the dump). Note: to include new test
       data in the regression tests, edit psana_test/data/regressionTests.txt BEFORE running this.
       Otherwise delete the line for that test file in previousDump.txt and re-run this command.

       This file is used by the test command to detect changes.

       optional args:  
         delete=False    don't delete dump file (they will be in psana_test/data/test_output/$SIT_ARCH/prev_xtc_dump)
         all=True        redo dump of all tests - not just new test files. 
                         Does not append, creates psana_test/data/previousDump.txt from scratch.

links  creates new soft links in $SIT_ROOT/data_test/types that link to the xtc in ../Translator
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
         set=xtc:n   or xtc:n,m,k    just do those regression test numbers for xtc files
         set=multi:n or multi:n,m,k  just regression test 0 from multi test set files
         set=full                    not the regression, processes complete xtc of all tests
         set=multi:full or xtc:full  limit full to that dataset
         set=multi:full:n or multi:full:n,m,k   a select number of full tests to do

testshm test psana shared memory datasource.

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
    def __init__(self,msg=''):
        super(Alarm, self).__init__(msg)

def alarm_handler(signum, frame):
    raise Alarm

###########################################################
# testing functions
def cmdTimeOutWithReturnCode(cmd,seconds=5*60):
    '''Runs cmd with seconds timeout - raising exceptoin Alarm if timeout is hit.
    warning: runs cmd with shell=True, considered a security hazard.
    Returns stdout, stderr, returncodefor command. stderr has had blank links removed.
    '''
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(seconds)
    try:
        p = sb.Popen(cmd, shell=True, stdout=sb.PIPE, stderr=sb.PIPE)
        o,e = p.communicate()        
        signal.alarm(0)  # reset the alarm
    except Alarm:
        raise Alarm("cmd: %s\n took more than %d seconds" % (cmd, seconds))
    e = [ln for ln in e.split('\n') if len(ln.strip())>0]
    return o, '\n'.join(e), p.returncode
    
def cmdTimeOut(cmd,seconds=5*60):
    '''Runs cmd with seconds timeout - raising exceptoin Alarm if timeout is hit.
    warning: runs cmd with shell=True, considered a security hazard.
    Returns stdout, stderr for command. stderr has had blank links removed.
    '''
    o,e,retcode = cmdTimeOutWithReturnCode(cmd,seconds)
    return o,e
    
def expandSitRoot():
    sit_root_envvar = '$SIT_ROOT'
    sit_root = os.path.expandvars(sit_root_envvar)
    assert sit_root != sit_root_envvar, "%s not expanded. run sit_setup" % sit_root_envvar
    return sit_root

def expandSitArch():
    sit_arch_envvar = '$SIT_ARCH'
    sit_arch = os.path.expandvars(sit_arch_envvar)
    assert sit_arch != sit_arch_envvar, "%s not expanded. run sit_setup" % sit_arch_envvar
    return sit_arch

def getDataArchDir(pkg, datasubdir, archsubdir=''):
    assert len(pkg)>0
    assert len(datasubdir)>0

    def makeIfDoesntExist(dirpath):
        if not os.path.exists(dirpath):
            try:
                os.mkdir(dirpath)
            except OSError:
                pass
            assert os.path.exists(dirpath), "failed to make directory: %s" % dirpath

    dataPkgDir  = AppDataPath(pkg).path()
    assert os.path.exists(dataPkgDir), "package directory: %s doesn't exist" % dataPkgDir
    subDir = os.path.join(dataPkgDir, datasubdir)
    makeIfDoesntExist(subDir)
    sit_arch = expandSitArch()
    finalDir = os.path.join(subDir, sit_arch)
    makeIfDoesntExist(finalDir)
    if len(archsubdir) > 0:
        finalDir = os.path.join(finalDir, archsubdir)
        makeIfDoesntExist(finalDir)
    return finalDir

def instrDataDir():
    return '/reg/d/psdm'

def getTestDataDir():
    testDataDir = os.path.join(expandSitRoot(), 'data_test/Translator')
    assert os.path.exists(testDataDir), "test data dir does not exist." + \
        " Is this an external distrubtion? test data not available outside SLAC"
    return testDataDir

def getTestCalibDir():
    testCalibDir = os.path.join(expandSitRoot(), 'data_test/calib')
    assert os.path.exists(testCalibDir), "test calib dir does not exist." + \
        " Is this an external distrubtion? test data not available outside SLAC"
    return testCalibDir

def getMultiFileDataDir():
    multiDataDir = os.path.join(expandSitRoot(), 'data_test/multifile')
    assert os.path.exists(multiDataDir), "multifile test data dir does not exist." + \
        " Is this an external distrubtion? test data not available outside SLAC"
    return multiDataDir

def getTestFiles(noTranslator=False):
    '''returns the test files as a dictionary. These are the files
    of the form test_xxx_*.xtc where xxx is an integer. These are the single
    xtc files, not the multi file datasets.

    optional arg: noTranslator=True means other than test 42, don't 
    return tests with Translator in the name.

    Returns a dict:
      res[n]={'basename':basename, 'path':path}
    where n is the testnumber, 
          basename  is the file basename, such as test_000_exp.xtc
          path      is the full path to the xtc file.
    '''
    testDataDir = getTestDataDir()
    files = glob.glob(os.path.join(testDataDir,'test_*.xtc'))
    assert len(files)>0, "no test_*.xtc files found in directory: %s" % testDataDir
    basenames = [os.path.basename(fname) for fname in files]
    numbers = [int(basename.split('_')[1]) for basename in basenames]
    res = {}
    for num, basename, fullpath in zip(numbers, basenames, files):
        if noTranslator:
            if num != 42 and basename.find('Translator')>=0:
                continue
        res[num] = {'basename':basename, 'path':fullpath}
    return res

def getMultiDatasets():
    '''returns the multi file datasets as a dictionary. 
    If at the base directory we have

    ./test_000_amo01509
    ./test_000_amo01509/e8-r0125-s00-c00.xtc
    ./test_000_amo01509/e8-r0125-s01-c00.xtc

    Then the dict will be (assuming $SIT_ROOT = /reg/g/psdm):
    
    res[0]={'basedir':'test_000_amo01509',
            'basepath':'/reg/g/psdm/data_test/multifile/test_000_amo01509',
            'xtcs':['e8-r0125-s00-c00.xtc','e8-r0125-s01-c00.xtc']
            'runs':[125]
            'dspec':'exp=amo01509:run=125:dir=/reg/g/psdm/data_test/multifile/test_000_amo01509'}

    The fields are for the most part self explanatory. dspec will always be a psana dataset
    specification to process all the runs in the directory.
    '''
    multiBaseDir = getMultiFileDataDir()
    assert os.path.exists(multiBaseDir), "multifile directory does not exist: %s" % multiBasedir
    multiDirPaths = glob.glob(os.path.join(multiBaseDir, 'test_*'))
    res = {}
    for multiDirPath in multiDirPaths:
        multiDir = os.path.basename(multiDirPath)
        number = int(multiDir.split('_')[1])
        xtcPaths = glob.glob(os.path.join(multiDirPath,"*.xtc"))
        xtcFiles = []
        runs = set()
        for xtcFilePath in xtcPaths:
            xtcFile = os.path.basename(xtcFilePath)
            flds = xtcFile.split('-')
            assert len(flds)==4, "xtcFile %s doesn't have 4 fields separated by '-'" % xtcFile
            assert flds[0].startswith('e'), "xtcFile %s doesn't start with e" % xtcFile
            assert flds[1].startswith('r'), "xtcFile %s doesn't have -r in fld1" % xtcFile
            assert flds[2].startswith('s'), "xtcFile %s doesn't have -s in fld2" % xtcFile
            assert flds[3].startswith('c'), "xtcFile %s doesn't have -c in fld3" % xtcFile
            runs.add(int(flds[1][1:]))
            xtcFiles.append(xtcFile)
        runs = list(runs)
        runs.sort()
        expname = '_'.join(multiDir.split('_')[2:])
        res[number] = {'basedir': multiDir,
                       'basepath': multiDirPath,
                       'xtcs': xtcFiles,
                       'runs':runs,
                       'dspec':'exp=%s:run=%s:dir=%s' % (expname, ','.join(map(str,runs)), multiDirPath)}
    return res

def getPreviousDumpFilename():
    return os.path.join('psana_test', 'data', 'previousDump.txt')

def getRegressionTestFilename():
    return os.path.join('psana_test','data','regressionTests.txt')

def getRegressionTestExpectedDifferencesFilename():
    return os.path.join('psana_test','data','regressionTestsExpectedDiffs.xml')

def getRegressionTestExpectedDifferences():
    tree = ET.parse(getRegressionTestExpectedDifferencesFilename())
    return parseXmlTreeForExpectedDiffs(tree)

def getFullTestExpectedDifferencesFilename():
    return os.path.join('psana_test','data','fullTestsExpectedDiffs.xml')

def getFullTestExpectedDifferences():
    tree = ET.parse(getFullTestExpectedDifferencesFilename())
    return parseXmlTreeForExpectedDiffs(tree)

def parseXmlTreeForExpectedDiffs(tree):
    root = tree.getroot()
    expectedDiffs = {'xtc':{},'multi':{}}
    for regress in root.findall('regress'):
        assert regress.attrib['src'] in ['xtc','multi'], "src attribute is not in xtc or multi"
        if regress.attrib['src']=='xtc':
            expectedDiffs['xtc'][int(regress.attrib['number'])]=regress.find('diff').text.strip()
        elif regress.attrib['src']=='multi':
            expectedDiffs['multi'][int(regress.attrib['number'])]=regress.find('diff').text.strip()
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
    IN: fname - fullname
    OUT: text string - md5sum.
    '''
    assert os.path.exists(fname), "filename %s does not exist" % fname
    cmd = 'md5sum %s' % fname
    if verbose: print cmd
    secondsTimeOut = 15*60
    t0=time.time()
    o,e = cmdTimeOut(cmd, seconds=secondsTimeOut)
    totalTime = time.time()-t0
    assert len(e.strip())==0, "cmd: %s produced errors. secondsTimeOut=%s and total time was %s\nerrors:\n%s" % (cmd, secondsTimeOut, totalTime, e)
    flds = o.split()
    assert len(flds)==2, "output of md5 did not have 2 fields, output is: %s" % o
    assert flds[1] == fname, "output of md5 did not have filename in second field, flds[1]=%s, while filename=%s and flds[0]=%s" % (flds[1], fname, flds[0])
    return flds[0]

def psanaDump(inDataset, outfile, events=None, dumpEpicsAliases=False, regressDump=True, verbose=False, dumpBeginJobEvt=True):
    '''Runs the  psana_test.dump module on the inDataset and saves the output
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
    beginJobEvtStr = ''
    if not dumpBeginJobEvt:
        beginJobEvtStr = '-o psana_test.dump.dump_beginjob_evt=False'
    cmd = 'psana  -c "" %s -m psana_test.dump %s %s %s %s' % (numEventsStr, epicAliasStr, regressDumpStr, beginJobEvtStr, inDataset)
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
        prevXtc = set()
        prevMulti = set()
    else:
        assert os.path.exists(prevFullName), "file %s does not exist, run with all=True" % prevFullName
        prevXtcDict, prevMultiDict = readPrevious()
        prevXtc = prevXtcDict.keys()
        prevMulti = prevMultiDict.keys()
        fout = file(prevFullName,'a')
    testTimes = {'xtc':{}, 'multi':{}}
    regressTests = readRegressionTestFile()
    print "** prev: carrying out md5 sum of xtc, psana_test.dump of xtc and regression tests."
    print "   (dump a few events for some xtc, without epics aliases in dump, and no epics pvId's in dump)"
    testFiles = getTestFiles()
    multiTests = getMultiDatasets()
    dumpOutputDir = getDataArchDir(pkg='psana_test', datasubdir='test_output', archsubdir='prev_xtc_dump')
    for src,srcDict in zip(['xtc','multi'],[testFiles, multiTests]):
        for testNumber, info in srcDict.iteritems():
            if (src == 'xtc') and (testNumber in prevXtc): continue
            if (src == 'multi') and (testNumber in prevMulti): continue
            t0 = time.time()
            xtc_md5s = {}
            if src == 'xtc':
                inputDatasource = info['path']
                xtc_md5s[info['basename'] ] = get_md5sum(inputDatasource)
                dumpOutput = os.path.join(dumpOutputDir, 'xtc_' + info['basename'] + '.dump')
                regressOutput = os.path.join(dumpOutputDir, 'regress_xtc_' + info['basename'] + '.dump')
            elif src == 'multi':
                inputDatasource = info['dspec']
                for xtc in info['xtcs']:
                    xtc_md5s[xtc] = get_md5sum(os.path.join(info['basepath'], xtc))
                dumpOutput = os.path.join(dumpOutputDir, 'multi_' + info['basedir'] + '.dump')
                regressOutput = os.path.join(dumpOutputDir, 'regress_multi_' + info['basedir'] + '.dump')
            cmd, err = psanaDump(inputDatasource, dumpOutput, dumpEpicsAliases=True, regressDump=False)
            if len(err)>0:
                fout.close()
                errMsg = '** FAILURE ** createPrevious: psana_test.dump produced errors on test %d\n' % testNumber
                errMsg += 'cmd: %s\n' % cmd
                errMsg += err
                raise Exception(errMsg)
            dump_md5 = get_md5sum(dumpOutput)
            if deleteDump: os.unlink(dumpOutput)
            regress_md5 = '0' * 32
            if testNumber in regressTests[src]:
                testInfo = regressTests[src][testNumber]
                regressInput, events, dumpEpicsAliases = testInfo['datasource'], testInfo['events'], \
                                                       testInfo['dumpEpicsAliases']
                if src == 'xtc':
                    assert regressInput == inputDatasource, "xtc file regression test %d xtc != testData xtc, regress=%s testData=%s" % \
                        (testNumber, regressInput, inputDatasource)
                elif src == 'multi':
                    # see if there is a dir= on the datasource and that it matches the testinfo
                    # datasource
                    assert len(inputDatasource.split('dir='))==2, "There is no dir= on the testData"
                    inputDir = inputDatasource.split('dir=')[1]
                    failMsg = "regress multi test %d - regress datasource != testData dir\n" % testNumber
                    failMsg += "regress specifies: %s\n searching test data found: %s" % \
                               ( testInfo['datasource'], inputDir)
                    assert inputDir == testInfo['datasource'], failMsg

                cmd, err = psanaDump(inputDatasource, regressOutput, events, dumpEpicsAliases=dumpEpicsAliases, 
                                     regressDump=True)
                if len(err)>0:
                    fout.close()
                    errMsg = '** FAILURE ** createPrevious: psana_test.dump produced errors on regress test %d\n' % testNumber
                    errMsg += 'cmd: %s\n' % cmd
                    errMsg += err
                    raise Exception(errMsg)
                regress_md5 = get_md5sum(regressOutput)
                if deleteDump: os.unlink(regressOutput)
            if src == 'xtc':
                fout.write("xtc_md5sum=%s   dump_md5sum=%s   regress_md5sum=%s   xtc=%s\n" % \
                           (xtc_md5s.values()[0], dump_md5, regress_md5, info['basename']))
            elif src == 'multi':
                fout.write("xtc_md5sum=%s   dump_md5sum=%s   regress_md5sum=%s   multi=%s" % \
                           ('0' * 32, dump_md5, regress_md5, info['basedir']))
                xtcs = xtc_md5s.keys()
                xtcs.sort()
                for xtc in xtcs:
                    md5 = xtc_md5s[xtc]
                    fout.write(" %s_md5sum=%s" % (xtc, md5))
                fout.write('\n')
            fout.flush()
            testTimes[src][testNumber] = time.time()-t0
            print "** prev:  %s test=%3d time=%.2f seconds" % (src, testNumber, testTimes[src][testNumber])
    fout.close()
    testsTime = sum(testTimes['xtc'].values())+sum(testTimes['multi'].values())
    print "* tests time: %.2f sec, or %.2f min" % (testsTime,testsTime/60.0)

def readPrevious():
    '''This parses the file of previous md5 resuls for the 
    test data, a psana_test.dump of the test data, and possibly 
    a psana_test.dump of a regression test of the test data.
    Regression tests may not run on all of the input and use special
    parameters for the dump.

    There are two kinds of test data. Single .xtc files, and 
    multifile datasets. A record of single file data looks like:

    xtc_md5sum=... dump_md5sum=... regress_md5sum=... xtc=...

    A record for multifile data looks like

    xtc_md5sum=... dump_md5sum=... regress_md5sum=... multifile=... filename1.xtc_md5sum=... filename2.xtc_md5sum=...

    For a signle xtc file, the four fields record:

    xtc_md5sum      md5 sum of the xtc file
    dump_md5sum     md5 sum of running psana_test.dump with no parameters on the whole xtc file
    regress_md5sum  md5 sum of regression test output for this xtc file, or 0 if not part of regression tests
    xtc             base name of the xtc file (not the full path).

    For a multifile dataset, the fields are

    xtc_md5sum      0 not used for multifile
    dump_md5sum     md5sum of the dump on the entire dataset, all runs in consecutive order
    regress_md5sum= md5sum of fthe dump of the regression test
    multifile= directory name for the dataset
    # next the md5sum's and filenames of all the files in the dataset are recorded.
    # the filenames are part of the keys, the md5sum is the value
    filename1.xtc_md5sum= the md5sum of the file named filename1.xtc
    filename2.xtc_md5sum= likewise the md5sum of the file named filename2.xtc

    What is returned is two dictionaries. One for the individual xtc test files, and
    another for the multifile test sets.
    
    xtcPrev, multiPrev

    The keys will be test numbers (there is one set of test numbers for xtc files, and another
    for multifile test sets). Where the keys are:

    xtcPrev[0]['xtc'] = xtc file basename
    xtcPrev[0]['md5xtc'] = md5 of xtc
    xtcPrev[0]['md5dump'] = md5 of dump of xtc
    xtcPrev[0]['md5regress'] = md5 dump of regress

    for multi, 
    xtcPrev[0]['xtcs'] = ['filename1', 'filename2', ...]
    xtcPrev[0]['md5xtcs']['filename1'] = md5 of filename1, 
    xtcPrev[0]['md5xtcs']['filename2'] = md5 of filename2, 
    ...
    xtcPrev[0]['md5dump'] = md5 of dump of whole dataset
    xtcPrev[0]['md5regress'] = md5 of regress test dump output
    '''

    def parseXtcLine(ln):
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
        return number, xtc, md5xtc, md5dump, md5regress

    def parseMultiLine(ln):
        ln,multiAndXtcs = ln.split(' multi=')
        ln,md5regress = ln.split(' regress_md5sum=')
        ln,md5dump = ln.split('dump_md5sum=')
        ln,md5xtc = ln.split('xtc_md5sum=')
        md5xtcs={}
        flds = multiAndXtcs.rsplit('_md5sum=',1)
        while len(flds)==2:
            md5sum = flds[1].strip()
            nextFlds = flds[0].split()
            assert len(nextFlds)>=2, "expecting 'm5sumvalue filename' (with space) but got %s" % flds[0]
            xtc = nextFlds[-1].strip()
            md5xtcs[xtc]=md5sum
            multiAndXtcs = ' '.join(nextFlds[0:-1])
            flds = multiAndXtcs.rsplit('_md5sum=',1)
        xtcs = md5xtcs.keys()
        multi = multiAndXtcs.strip()
        assert multi.startswith('test_'), "multi name doesn't start with test_: %s" % multi
        number = int(multi.split('_')[1])
        md5regress = md5regress.strip()
        md5dump = md5dump.strip()
        md5xtc = md5xtc.strip()
        return number, xtcs, md5xtcs, md5dump, md5regress

    prevFilename = getPreviousDumpFilename()
    assert os.path.exists(prevFilename), "file %s doesn't exist, run prev command" % prevFilename
    xtcRes = {}
    multiRes = {}
    for ln in file(prevFilename).read().split('\n'):
        if not ln.startswith('xtc_md5sum='):
            continue
        if ln.find(' xtc=') >=0:
            number, xtc, md5xtc, md5dump, md5regress = parseXtcLine(ln)
            xtcRes[number]={'xtc':xtc, 'md5xtc':md5xtc, 'md5dump':md5dump, 'md5regress':md5regress}
        elif ln.find(' multi=') >= 0:
            number, xtcs, md5xtcs, md5dump, md5regress = parseMultiLine(ln)
            multiRes[number]={'xtcs':xtcs, 'md5xtcs':md5xtcs, 'md5dump':md5dump, 'md5regress':md5regress}
    return xtcRes, multiRes

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
                lnk = os.path.join(expandSitRoot(),
                                   'data_test/types/%s.xtc' % tpForFileName)
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
                lnk = os.path.join(expandSitRoot(), 'data_test/types/%s.xtc' % epicsLnk)
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
    for fname in glob.glob(os.path.join(expandSitRoot(),'data_test/types/epicsPv*.xtc')):
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

testShmDumpTypes=set(['type=psana.Bld.BldDataEBeamV3, src=BldInfo(EBeam)',
                      'type=psana.Bld.BldDataFEEGasDetEnergy, src=BldInfo(FEEGasDetEnergy)',
                      'type=psana.Bld.BldDataPhaseCavity, src=BldInfo(PhaseCavity)',
                      'type=psana.ControlData.ConfigV2, src=ProcInfo(0.0.0.0, pid=27455)',
                      'type=psana.Epics.ConfigV1, src=DetInfo(EpicsArch.0:NoDevice.0)',
                      'type=psana.EvrData.ConfigV7, src=DetInfo(NoDetector.0:Evr.0)',
                      'type=psana.EvrData.ConfigV7, src=DetInfo(NoDetector.0:Evr.1)',
                      'type=psana.EvrData.DataV3, src=DetInfo(NoDetector.0:Evr.0)',
                      'type=psana.Ipimb.ConfigV2, src=BldInfo(XppSb2_Ipm)',
                      'type=psana.Ipimb.ConfigV2, src=BldInfo(XppSb3_Ipm)',
                      'type=psana.Ipimb.DataV2, src=BldInfo(XppSb2_Ipm)',
                      'type=psana.Ipimb.DataV2, src=BldInfo(XppSb3_Ipm)',
                      'type=psana.Lusi.IpmFexConfigV2, src=BldInfo(XppSb2_Ipm)',
                      'type=psana.Lusi.IpmFexConfigV2, src=BldInfo(XppSb3_Ipm)',
                      'type=psana.Lusi.IpmFexV1, src=BldInfo(XppSb2_Ipm)',
                      'type=psana.Lusi.IpmFexV1, src=BldInfo(XppSb3_Ipm)'])

def testShmCommand(args):
    ''' it is reccommended that you increase the message size from
    10 to 32 before running this test:
    sudo /sbin/sysctl -w fs.mqueue.msg_max=32
    '''
    shmemName = 'psana_test'
    testFile = os.path.join(getTestDataDir(),'test_042_Translator_t1.xtc')
    assert os.path.exists(testFile), "testfile not found: %s" % testFile
    xtcservercmd = 'xtcmonserver -f %s -p %s' % (testFile, shmemName)
    numberOfBuffers = 1
    xtcservercmd += ' -n %d' % numberOfBuffers
    sizeOfBuffers = '0x1000000'
    xtcservercmd += ' -s %s' % sizeOfBuffers
    ratePerSecond = 1
    xtcservercmd += ' -r %d' % ratePerSecond
    numberOfClients = 1
    xtcservercmd += ' -c %d' % numberOfClients
    OUTDIR = getDataArchDir(pkg='psana_test', datasubdir='test_output')
    xtcserverCmdStdout = os.path.join(OUTDIR,'testShm.xtcserver.stdout')
    xtcserverCmdStderr = os.path.join(OUTDIR,'testShm.xtcserver.stderr')
    dataSource = 'shmem=%s.0' % shmemName
    dumpcmd = 'psana -m psana_test.dump %s' % dataSource
    print "about to launch commands:"
    print xtcservercmd
    print dumpcmd
    expectedSharedMemoryFile = "/dev/shm/PdsMonitorSharedMemory_%s" % shmemName
    if os.path.exists(expectedSharedMemoryFile):
        print "ERROR: shared memory file exists: %s" % expectedSharedMemoryFile
        print " delete (if safe) and rerun test"
        return
    outputFiles = [xtcserverCmdStderr, xtcserverCmdStdout]
    for outputFile in outputFiles:
        if os.path.exists(outputFile):
            print "warning: test output file exists. deleting: %s" % outputFile
            os.unlink(outputFile)
        
    xtcservercmd += ' > %s 2>%s &' % (xtcserverCmdStdout, xtcserverCmdStderr)
    # run server in background while running dumpcmd
    print "running server cmd in bkgnd"
    try:
        os.system(xtcservercmd)
    except Exception,e:
        print "ERROR running server command."
        if os.path.exists(expectedSharedMemoryFile):
            print "deleteing shared memory server file: %s" % expectedSharedMemoryFile
            os.unlink(expectedSharedMemoryFile)
        raise e
    time.sleep(.35) # sleep a little in case the server needs time to start up    
    print "running dump cmd"
    try:
        dumpStdout,dumpError = cmdTimeOut(dumpcmd,30)
    except Alarm, alarm:
        print "ERROR: psana_test.dump command timed out: %s" % alarm
        print "  run ps and clean up jobs"
        print "  try to run test again, or try to increase the message size "
        print "  from 10 to 32 before running this test with the command: "
        print "  sudo /sbin/sysctl -w fs.mqueue.msg_max=32"
        print "===== xtcserver cmd stderr ======="
        print file(xtcserverCmdStderr).read()
        print "===== xtcserver cmd stdout ======"
        print file(xtcserverCmdStdout).read()
        if os.path.exists(expectedSharedMemoryFile):
            print "deleteing shared memory server file: %s" % expectedSharedMemoryFile
            os.unlink(expectedSharedMemoryFile)
        raise alarm

    # check dump outout
    print "===== dump stderr ====="
    print dumpError
    print "===== test result ====="
    typeLines = set()
    for ln in dumpStdout.split('\n'):
        if ln.startswith('type='):
            typeLines.add(ln.strip())
    missingTypeLines = testShmDumpTypes.difference(typeLines)
    if len(missingTypeLines)>0:
        print "ERROR: dump output does not list the following types: %r"  % missingTypeLines
    else:
        print "SUCCESS! shmem test passed. dump against shared memory appears to have produced expected output"
    
    time.sleep(.5)  # sleep a bit in case the server is not finished
    toDelete = [expectedSharedMemoryFile] + outputFiles    
    for fname in toDelete:
        if os.path.exists(fname):
            os.unlink(fname)

def translate(inDataset, outfile, numEvents, testLabel, verbose):
    numEventsStr = ''
    if numEvents is not None and numEvents > 0:
        numEventsStr = ' -n %d' % numEvents
    cmd = 'psana %s -m Translator.H5Output -o Translator.H5Output.output_file=%s -o Translator.H5Output.overwrite=True %s'
    cmd %= (numEventsStr, outfile, inDataset)
    if verbose: print cmd
    o,e = cmdTimeOut(cmd,20*60)
    e = '\n'.join([ ln for ln in e.split('\n') if not filterPsanaStderr(ln)])
    if len(e) > 0:
        errMsg =  "**Failure** %s: Translator failure\n" % testLabel
        errMsg += "cmd=%s\n" % cmd
        errMsg += "\n%s" % e
        raise Exception(errMsg)
    if verbose: print "%s: translation finished, produced: %s" % (testLabel, outfile)

def testCommand(args):
    # helper functions
    def checkForSameXtcFiles(testDataInfo, prevInfo, src, num, verbose):
        assert src in ['xtc','multi'], "unknown src: %s" % src
        if src == 'xtc':
            failMsg = "src=%s Test no=%d xtc mismatch: previous_dump.txt says "
            failMsg += "xtc files is:\n %s\n but for files on disk, it is:\n %s"
            failMsg %= (src, num, prevInfo['xtc'], testDataInfo['basename'])
            assert testDataInfo['basename'] == prevInfo['xtc'], failMsg
            if verbose: 
                msg = "src=%s test %d: xtc filename is the same as what was "
                msg += "recorded in previous_dump.txt. old=%s new=%s"
                msg %= (src, num, prevInfo['xtc'], testDataInfo['basename'])
                print msg
        elif src == 'multi':
            prevXtcs = prevInfo['xtcs']
            prevXtcs.sort()
            testXtcs = testDataInfo['xtcs']
            testXtcs.sort()
            lenFailMsg = "src=%s test=%d, %d != %d, that is previous number of xtc"
            lenFailMsg += " files in test != current number of xtc files in test"
            lenFailMsg %= (src, num, len(prevXtcs), len(testXtcs))
            assert len(prevXtcs)==len(testXtcs), lenFailMsg
            for prevXtc, testXtc in zip(prevXtcs, testXtcs):
                xtcFailMsg = "src=%s test=%d, xtc files differ: %s != %s"
                xtcFailMsg %= (src, num, prevXtc, testXtc)
                assert prevXtc == testXtc, xtcFailMsg
            if verbose:
                msg = "src=%s test %d: all xtc filenames the same"
                msg %= (src, num)
                print msg

    def checkForSameMd5sOfXtcFiles(testDataInfo, prevInfo, src, num, verbose):
        assert src in ['xtc','multi'], "unknown src: %s" % src
        if src == 'xtc':
            assert prevInfo['xtc'] == testDataInfo['basename']
            prevMd5s = [prevInfo['md5xtc']]
            testXtcs = [testDataInfo['path']]
        elif src == 'multi':
            prevXtcs = prevInfo['xtcs']
            prevXtcs.sort()
            testXtcs = testDataInfo['xtcs']
            testXtcs.sort()
            testXtcs = [os.path.join(testDataInfo['basepath'], xtc) for xtc in testXtcs]
            prevMd5s = [prevInfo['md5xtcs'][xtc] for xtc in prevXtcs]
        assert len(testXtcs)==len(prevMd5s)
        for testXtc, prevMd5 in zip(testXtcs, prevMd5s):
            current_md5 = get_md5sum(testXtc)
            failMsg = "src=%s test=%d md5 of xtc file has changed.\n"
            failMsg += "xtc=%s\n"
            failMsg += "prevMd5=%s\n"
            failMsg += "currMsg=%s\n"
            failMsg %= (src, num, testXtc, prevMd5, current_md5)
            assert current_md5 == prevMd5, failMsg
        if verbose:
            msg = "src=%s test=%d: calculated md5 of xtc files."
            msg += " They agree with previously recorded md5s"
            msg %= (src, num)
            print msg

    def compareXtcH5Dump(currentXtcDumpPath, h5DumpPath, testLabel, verbose, expectedOutput=''):
        cmd = 'diff %s %s' % (currentXtcDumpPath, h5DumpPath)
        if verbose: print cmd
        o,e = cmdTimeOut(cmd,5*60)
        assert len(e)==0, "** FAILURE running cmd: %s\nstder:\n%s" % (cmd,e)
        if o.strip() != expectedOutput:
            msg = "** FAILURE: diff in xtc dump and h5 dump %s not equal to expected output\n" % testLabel
            msg += " cmd= %s\n" % cmd
            msg += "-- diff output: --\n"
            msg += o
            if len(expectedOutput)>0:
                msg += "-- expected output: --\n"
                msg += expectedOutput
            raise Exception(msg)        
        if verbose: print "%s: compared dump of xtc and dump of h5 file" % testLabel

    def removeFiles(files):
        for fname in files:
            assert os.path.exists(fname)
            os.unlink(fname)

    ##################################
    # end helper functions
    delete,verbose,testSet = parseArgs(args,['delete','verbose', 'set'])
    assert delete.lower() in ['','false','true']
    assert verbose.lower() in ['','false','true']        

    dumpOutputDir = getDataArchDir(pkg='psana_test', datasubdir='test_output', archsubdir='current_xtc_dump')
    h5dir = getDataArchDir(pkg='psana_test', datasubdir='test_output', archsubdir='current_h5')
    h5dumpDir = getDataArchDir(pkg='psana_test', datasubdir='test_output', archsubdir='current_h5_dump')
    delete = not (delete.lower() == 'false')
    verbose = (verbose.lower() == 'true')
    xtcTestFiles = getTestFiles(noTranslator=True)
    multiTestSets = getMultiDatasets()
    testFiles = {'xtc':xtcTestFiles, 'multi':multiTestSets}
    prev = {'xtc':None, 'multi':None}
    prev['xtc'], prev['multi'] = readPrevious()
    regress = readRegressionTestFile()
    whichTest = 'regress'
    testNumberFilter = {'xtc':regress['xtc'].keys(), 'multi':regress['multi'].keys()}
    expectedDiffs = getRegressionTestExpectedDifferences()
    srcs = ['xtc','multi']
    if testSet.startswith('xtc:'):
        srcs = ['xtc']
        testSet = testSet.split('xtc:')[1]
    elif testSet.startswith('multi:'):
        srcs = ['multi']
        testSet = testSet.split('multi:')[1]
    if testSet.startswith('full'):
        whichTest = 'full'
        expectedDiffs = getFullTestExpectedDifferences()
        jnk, afterFull = testSet.split('full')
        if len(afterFull)>0:
            assert afterFull.startswith(':'), "must follow full with : to specify tests, not '%s' " % afterFull
            jnk, testSet = afterFull.split(':')
        else:
            testSet = ''
    if testSet != '':
        commaSepTestSet = testSet.split(',')
        testNumberFilter['xtc'] = []
        testNumberFilter['multi'] = []
        for val in commaSepTestSet:
            if val.find('-')>0:
                a,b = map(int,val.split('-'))
                for src in srcs:
                    testNumberFilter[src].extend(range(a,b+1))
            else:
                for src in srcs:
                    testNumberFilter[src].append(int(val))

    for src in srcs:
        for num in testNumberFilter[src]:
            if num in testFiles[src]:
                # for full, make sure every test number has a previous entry.
                # for regress, its ok to specify numbers that aren't in the regression set,
                # but if they are, make sure there is a previous entry
                if (whichTest == 'full') or (whichTest == 'regress' and num in regress[src]):
                    assert num in prev[src], "There is a new %s test number: %s\n. Run prev command first." % (src,num)

    if verbose:
        print "psanaTestLib - about to run %s tests:" % whichTest
        xtcTests = testNumberFilter['xtc']
        multiTests = testNumberFilter['multi']
        xtcTests.sort()
        multiTests.sort()
        if whichTest == 'regress':
            xtcTests = [x for x in xtcTests if x in regress['xtc']]
            multiTests = [x for x in multiTests if x in regress['multi']]
        print "  xtc: %s" % ','.join(map(str,xtcTests))
        print " multi: %s" % ','.join(map(str,multiTests))

    testTimes = {'xtc':{}, 'multi':{}}
    for src in srcs:
        for num in testNumberFilter[src]:
            if num not in testFiles[src]: continue
            t0 = time.time()
            testDataInfo = testFiles[src][num]
            prevInfo = prev[src][num]
            checkForSameXtcFiles(testDataInfo, prevInfo, src, num, verbose)            
            checkForSameMd5sOfXtcFiles(testDataInfo, prevInfo, src, num, verbose)
            # set parameters passed to psana_test.dump to default, for full test
            regressStr = ''
            numEvents = 0
            dumpEpicsAliases=True
            regressDump=False
            md5prev = prev[src][num]['md5dump']
            if whichTest == 'regress':
                if num not in regress[src]:
                    print "warning: src=%s test=%d not in regress tests, skipping" % (src,num)
                    continue
                regressStr = 'regress_'
                numEvents = regress[src][num]['events']
                dumpEpicsAliases=False
                regressDump=True
                md5prev = prev[src][num]['md5regress']
            if src == 'xtc':
                inputDataSource = testDataInfo['path']
                dumpOutputBase = regressStr + 'xtc_' + testDataInfo['basename']
                h5OutputBase = regressStr + 'xtc_' + testDataInfo['basename']
            elif src == 'multi':
                inputDataSource = testDataInfo['dspec']
                dumpOutputBase = regressStr + 'multi_' + testDataInfo['basedir']
                h5OutputBase = regressStr + 'multi_' + testDataInfo['basedir']
            dumpOutputBase += '.dump'
            h5OutputBase = os.path.splitext(h5OutputBase)[0] + '.h5'
            h5OutputDumpBase = h5OutputBase + '.dump'
            testLabel = "%s%s_test_%3.3d" % (regressStr, src, num)
            dumpOutputPath = os.path.join(dumpOutputDir, dumpOutputBase)
            cmd, err = psanaDump(inputDataSource, dumpOutputPath, 
                                 numEvents, dumpEpicsAliases=dumpEpicsAliases, 
                                 regressDump=regressDump, verbose=verbose)
            if len(err) > 0:
                raise Exception("**Failure: %s, psanaDump call failed.\n cmd=%s\n%s" % (testLabel, cmd, err))
           
            md5current = get_md5sum(dumpOutputPath)
            failMsg = "**FAIL: %s" % testLabel
            failMsg += " md5 of dump does not agree with previously recorded\n."
            failMsg += "cmd=%s\n" % cmd
            failMsg += "prev_md5=%s\n" % md5prev
            failMsg += "curr_md5=%s" % md5current
            assert md5prev == md5current, failMsg

            if verbose: 
                msg = "%s%s test=%d: success: same md5 for xtc data as previously recorded" 
                msg %= (regressStr, src, num)
                print msg

            h5file = os.path.join(h5dir, h5OutputBase)
            translate(inputDataSource, h5file, numEvents, testLabel, verbose)
            h5DumpPath = os.path.join(h5dumpDir, h5OutputDumpBase)
            cmd, err = psanaDump(h5file, h5DumpPath, numEvents, 
                                 dumpEpicsAliases=dumpEpicsAliases, regressDump=regressDump, verbose=verbose)
            if len(err) > 0:
                raise Exception("**Failure: test=%d, psanaDump failed on h5.\n cmd=%s\n%s" % (num, cmd, err))
            # skip h5 vs xtc dump compare for full
            if whichTest == 'regress':
                xtc2h5ExpectedDiff = expectedDiffs[src].get(num,'')
                compareXtcH5Dump(dumpOutputPath, h5DumpPath, num, verbose, expectedOutput=xtc2h5ExpectedDiff)
            if delete:
                removeFiles([dumpOutputPath, h5DumpPath, h5file])
            testTime = time.time()-t0
            testTimes[src][num]=testTime
            print "%s success: total time: %.2f sec or %.2f min" % (testLabel,testTime,testTime/60.0)

    totalTestTimes = sum(testTimes['xtc'].values()) + sum(testTimes['multi'].values())
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
    newTestFilePath = os.path.join(expandSitRoot(),'data_test/Translator',newTestFileBaseName)
    bytesToCopy = max([el[1] for el in l1acceptFollowing.values()])
    copyBytes(xtc, bytesToCopy, newTestFilePath)

def copyToMultiTestDir(experiment, run, numberCalibCycles, numberEventsPerCalibCycle, 
                       destDir, fiducialList=None, index=True):
    '''copies specified number of events/calib cycles from regular data to 
    destination directory. Optionally copies datagrams matching the given fidicuals as well.
    only copies the first occurence of any fiducial.
    INPUT
     experiment, run  - for example experiment=xppd7114 and run=130 means that the
                        all streams and chunks of those xtc files in 
                        /reg/g/psdm/data will be examined.
    numberCalibCycles - how many calib cycles to copy events from
    numberEventsPerCalibCycle - how many events per calib cycle *FOR EACH STREAM * to copy out
                                that is this is not a total for the calib cycle
    destDir  - the destination directory for the output
    fiducialList - for example [34263, 34266, 35436, 36009, 44559, 54090, 54696, 54699]
                   means that datagrams with those fiducials will be copies out as well
                   (when they first occur) however if they occur past the last calib cycle
                   they will not be copied
    index - make the index files afterwards
    '''
    def getEndOffset(lns, lnIdx, xtc):
        '''helper function.
        INPUT
        lns   - list of lines from xtclinedump output
        lnIdx - index of line in lns that we want end offset of
        xtc   - name of xtc file for which this is a xtclinedump
        OUTPUT
        the offset in the xtc of the next byte after the datagram as lnIdx
        '''
        if lnIdx < len(lns)-1:
            nextDg = XtcLine(lns[lnIdx+1])
            if nextDg.contains == 'dg':
                endOffset = nextDg.offset
            else:
                endOffset = os.stat(xtc).st_size
        else:
            endOffset = os.stat(xtc).st_size
        return endOffset

    dataDir = os.path.join('/','reg','d','psdm', experiment[0:3], experiment, 'xtc')
    assert os.path.exists(dataDir), "data dir not found: %s" % dataDir
    assert os.path.exists(destDir), "dest dir not found: %s" % destDir
    xtcFiles = glob.glob(os.path.join(dataDir,'*-r%4.4d*.xtc'%run))
    xtcFiles.sort()
    chunk0 = [fname for fname in xtcFiles if fname.endswith('-c00.xtc')]
    chunkn = xtcFiles[-len(chunk0):]
    dgramQueue = ['Configure','BeginRun']
    if fiducialList is None:
        fiducialList = set()
    else:
        fiducialList = set(fiducialList)
    for cc in range(numberCalibCycles):
        dgramQueue.extend(['BeginCalibCycle','Enable'])
        for evt in range(numberEventsPerCalibCycle):
            dgramQueue.append('L1Accept')
        dgramQueue.extend(['Disable','EndCalibCycle'])
    dgramQueue.append('EndRun')
        
    for xtc0,xtcn in zip(chunk0,chunkn):
        done = False
        queueIdx = 0
        outFile = os.path.join(destDir,os.path.basename(xtc0))
        xtcs = [xtc0]
        if xtcn != xtc0:
            xtcs.append(xtcn)
        for xtcIdx,xtc in enumerate(xtcs):
            cmd = 'xtclinedump dg %s' % xtc
            o,e = cmdTimeOut(cmd,10*60)
            assert len(e)==0, "Failure running cmd%s\nError=%s" % (cmd,e)
            bytesToCopy=[]
            lns = o.split('\n')
            numberEndCalibCyclesSeen = 0
            for lnIdx,ln in enumerate(lns):
                if len(ln.strip())==0: continue
                dg = XtcLine(ln)
                assert dg.contains == 'dg', 'xtclinedump dg line not dg: ln= %s' % ln
                copiedDgram = False
                startOffset = dg.offset
                if queueIdx < len(dgramQueue) and dg.sv.strip() == dgramQueue[queueIdx]:
                    if dgramQueue[queueIdx]=='EndCalibCycle':
                        numberEndCalibCyclesSeen += 1
                    queueIdx += 1
                    endOffset = getEndOffset(lns, lnIdx, xtc)
                    bytesToCopy.append((startOffset,endOffset))
                    copiedDgram = True
                if (dg.fid in fiducialList) and \
                   (numberEndCalibCyclesSeen < numberCalibCycles) and (not copiedDgram):
                    endOffset = getEndOffset(lns, lnIdx, xtc)
                    bytesToCopy.append((startOffset, endOffset))
                    fiducialList.discard(dg.fid)
                if queueIdx == len(dgramQueue):
                    done = True
                    break
            if xtcIdx == 0:
                pass
                copyBytes(xtc,bytesToCopy,outFile,'truncate')
            else:
                pass
                copyBytes(xtc,bytesToCopy,outFile,'append')
            if done:
                break
    if index:
        indexDir = os.path.join(destDir, 'index')
        if not os.path.exists(indexDir):
            print "Creating index dir: %s" % indexDir
            os.mkdir(indexDir)
        xtcFiles = glob.glob(os.path.join(destDir,'*.xtc'))
        print "------------"
        for xtcFile in xtcFiles:
            outputFile = os.path.join(indexDir, os.path.basename(xtcFile) + '.idx')
            cmd = 'xtcindex -f %s -o %s' % (xtcFile, outputFile)
            print "running command: %s" % cmd
            o,e = cmdTimeOut(cmd)
            print "--output--\n%s\n--error--%s\n-------" % (o,e)
       
        
def copyBytes(src, n, dest, mode='truncate'):
    '''copies the fist n bytes, or a set of invervals, the intervals are 
    treated as [a,b), i.e, copyBytes(fileA,[[10,13]],fileB) copies bytes
    10,11 and 12 from fileA to make fileB.

    set mode to 'append' to append
    '''
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
    if mode == 'truncate':
        outFile = io.open(dest,'wb')
    elif mode == 'append':
        outFile = io.open(dest,'ab')
    else:
        raise ValueError("Mode is neither 'append' nor 'truncate' it is %s" % mode)
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
        instPath = os.path.join(instrDataDir(), instDir)
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
    regressTests = {'xtc':{},'multi':{}}
    for ln in fin:
        ln = ln.strip()
        if len(ln)==0: continue
        if ln.startswith('#'): continue
        if ln.find('xtc=')>0:
            ln,ds = ln.split(' xtc=')
            src = 'xtc'
        elif ln.find('multi=')>0:
            ln,ds = ln.split(' multi=')
            src = 'multi'
        else:
            raise Exception("regression test file contains line with neither xtc= or multi=, ln:\n%s" % ln)
        ln,noEpicsAliases=ln.split(' doNotDumpEpicsAliases=')
        ln,events=ln.split(' events=')
        ln,testno = ln.split('test=')
        events = int(events)
        noEpicsAliases = int(noEpicsAliases)
        testno = int(testno)
        regressTests[src][testno]={'events':events,'datasource':ds,'dumpEpicsAliases':not bool(noEpicsAliases)}
    return regressTests

cmdDict = {
    'prev':previousCommand,
    'links':makeTypeLinks,
    'test':testCommand,
    'testshm':testShmCommand,
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
