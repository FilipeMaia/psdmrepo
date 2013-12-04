import os
import sys
import collections
import math
import h5py

class MockTesterNoStop(object):
    def assertTrue(self,a,msg=''):
        if not a:
            print "**FAIL** assertTrue - a is false, msg=%s" % msg

    def assertEqual(self,a,b,msg=''):
        if a != b:
            print "**FAIL** assertEqual - a!=b\n  a=%r\n  b=%r\n  msg=%s" % (a,b,msg)
    
class MockTesterStop(object):
    def assertTrue(self,a,msg=''):
        assert a,"a is not true, msg=%s" % msg

    def assertEqual(self,a,b,msg=''):
        assert a==b,"a !=b, msg=%s, a=%r, b=%r" %(msg,a,b)

def getDistinctTimes(timeds):
    '''Takes a dataset with 'seconds' and 'nanoseconds' fields.
    Finds the unique pairs of these values.  Returns a sorted list of these
    distinct pairs in the dataset along with there positions in the timeds, for instance
    if
    timeds=array([(1364147551L, 107587445L, 331318L, 118410L, 140L, 0L),
                  (1364147551L, 174323092L, 331570L, 118434L, 12L, 6L),
                  (1364147551L, 174323092L, 331570L, 118434L, 12L, 6L)], 
      dtype=[('seconds', '<u4'), ('nanoseconds', '<u4'), ('ticks', '<u4'), ('fiducials', '<u4'), ('control', '<u4'), ('vector', '<u4')])

    then getDistinctTimes(timeds) would return:

    [ [1364147551L, 107587445L, [0] ],
      [1364147551L, 174323092L, [1,2] ]
    '''
    if len(timeds)==0:
        return []
    secs = timeds['seconds']
    nanos = timeds['nanoseconds']
    times = [(s,n,k) for s,n,k in zip(secs,nanos,range(len(timeds)))]
    times.sort()
    distinctTimes=[[times[0][0],times[0][1],[times[0][2]]]]
    for s,n,k in times[1:]:
        if distinctTimes[-1][0]==s and distinctTimes[-1][1]==n:
            distinctTimes[-1][2].append(k)
        else:
            distinctTimes.append([s,n,[k]])
    return distinctTimes                          

def compareTime(tester,o2oTime,psanaTime,diffs):
    o2oEventTimes = getDistinctTimes(o2oTime)
    psanaEventTimes = getDistinctTimes(psanaTime)
    o2oDistinct = [(s,n) for s,n,k in o2oEventTimes]
    psanaDistinct = [(s,n) for s,n,k in psanaEventTimes]
    common,o2oOnly,psanaOnly = partSets(o2oDistinct, psanaDistinct)
    if o2oDistinct != psanaDistinct:
        diffs['time dataset differs: o2o-only=%r psana-only=%r' % \
              (o2oOnly, psanaOnly)].add(o2oTime.name)
    commonTimes = list(common)
    commonTimes.sort()
    o2oIdx=[]
    psanaIdx=[]
    for s,n in commonTimes:
        while o2oEventTimes[0][0] != s and o2oEventTimes[0][1] != n:
            o2oEventTimes.pop(0)
        while psanaEventTimes[0][0] != s and psanaEventTimes[0][1] != n:
            psanaEventTimes.pop(0)
        o2oIdx.extend(o2oEventTimes[0][2])
        psanaIdx.extend(psanaEventTimes[0][2])
    return o2oIdx, psanaIdx
        
def partSets(A,B):
    A = set(A)
    B = set(B)
    inter = A.intersection(B)
    onlyA = A.difference(B)
    onlyB = B.difference(A)
    return inter,onlyA,onlyB

def attrDiffs(oA,oB):
    '''takes two h5py objects.  
    Return a 4-tuple:  diffs, onlyA, onlyB, dbgMsg
       where
    diffs - attributes that exist in both oA and oB but have different values.  
            The key names are returned, not the differing values.
    onlyA - those attribute keys only in oA.
    onlyB - sim
    dbgMsg - prints out the attriubres
    '''
    Akeys = oA.attrs.keys()
    Bkeys = oB.attrs.keys()
    Aset=set(Akeys)
    Bset=set(Bkeys)
    assert len(Aset)==len(Akeys)
    assert len(Bset)==len(Bkeys)    
    common,onlyA,onlyB = partSets(Aset,Bset)
    common_diffs = []
    debugMsg = "#### attributes: %s ####\n" % oA.name
    debugMsg += "common attributes\n"
    for ky in common:
        if oA.attrs.get(ky) != oB.attrs.get(ky):
            common_diffs.append(ky)
            debugMsg += " key differs: %s:  %s  vs.  %s\n" % (ky, oA.attrs.get(ky), oB.attrs.get(ky))
        else:
            debugMsg += " key is the same: %s = %s\n" % (ky, oA.attrs.get(ky))
    for ky in onlyA:
        debugMsg += " key only in A: %s" %ky
    for ky in onlyB:
        debugMsg += " key only in B: %s"%ky
    debugMsg += "\n######\n"
    return set(common_diffs),onlyA,onlyB,debugMsg

def groupKeys(o2o,psana,hdr):
    '''groups the keys between two h5py objects, returns
    common, only_o2o, only_psana, debugMsg
    '''
    common, only_o2o, only_psana = partSets(o2o.keys(), psana.keys())
    debugMsg = "####### %s ####\n%s\ncommon keys: %s\n" % (hdr, o2o.name,' '.join(common))
    debugMsg += "  o2o keys: %s\n" % ' '.join(only_o2o)
    debugMsg += "psana keys: %s\n" % ' '.join(only_psana)
    return common, only_o2o, only_psana, debugMsg

def compareTranslation(tester,o2o, psana, diffs, cmpDsetValues=True, recurse=True):
    '''High level function that recursively compares h5py objects from o2o-translate and
    the psana Translator.
    ARGS:
       tester          - an instance of a class that has the methods:
                         assertEqual(a,b,msg) 
                         assertTrue(a,msg)
                           such as unittest.TestCase
       o2o, psana      - the h5py objects. Typically, compareTranslation is initially called
                         with two h5py.File, but it recursively calls itself with sub objects.
       diffs           - a defaultdictionary with a set as the default (supports add method).
       cmpDsetValues   - defaults to True, which compare all values in the datasets. 
                         set to False to just compare the group hiearchy
       recurse         - defaults to True, compare like named subgroups.

    Tests that the hierarchies are the same.  Defaults to test that the datasets are the same.
    There are some acceptable differences between o2o-translate and psana.  
    These kinds of differences are not passed to the tester, they are recorded in the diffs 
    dictionary. Keys are string messages and values are lists of dataset names, 
    or h5 object names, where that difference occurred.
    '''
    attr_diffs, only_o2o_attr, only_psana_attr, attrDebugMsg = attrDiffs(o2o, psana)

    if type(o2o)!=h5py._hl.dataset.Dataset:
        common_keys, only_o2o_keys, only_psana_keys, keyDebugMsg  = groupKeys(o2o, psana,"sub-objects")

    if type(o2o)==h5py._hl.files.File:
        # make sure all common file attributes are the same except for the following:
        tester.assertEqual(attr_diffs,set([u'origin', u'UUID', u'created', u':schema:version']), 
                           msg="%s\nThe file attributes that are different are incorrect." % attrDebugMsg)

        # make sure the only file attributes in one or the other are the following:
        tester.assertEqual(only_o2o_attr, set(['runType','runNumber']), 
                           msg="%s\nfile attributes unique to o2o are wrong" % attrDebugMsg)
        tester.assertEqual(only_psana_attr, set(['instrument','jobName','expNum']), 
                           msg="%s\nfile attributes unique to psana are wrong" % attrDebugMsg)

        # make sure the file only has one group - Configure:0000
        tester.assertEqual(common_keys,set(['Configure:0000']), 
                           msg="%s\ncommon group from root is wrong. Should be Configure:0000.\n" % keyDebugMsg)
        tester.assertEqual(only_o2o_keys,set([]), 
                           msg="%s\no2o should have no keys that aren't common." % keyDebugMsg)
        tester.assertEqual(only_psana_keys,set([]), 
                           msg="%s\npsana should have no keys that aren't common." % keyDebugMsg)
        
        # recursively compare the configure group
        if recurse:
            compareTranslation(tester,o2o['Configure:0000'], psana['Configure:0000'], diffs, cmpDsetValues,recurse)

    elif type(o2o)!=h5py._hl.dataset.Dataset:
        # all common group attributes, save schema are the same
        tester.assertTrue((attr_diffs == set([]) or (attr_diffs == set(['_schemaVersion']))),
                          "%s\nthe only differing group attribute should be _schemaVersion" % attrDebugMsg)
        
        # report differences in schema version
        if attr_diffs == set(['_schemaVersion']):
            diffs['_schemaVersion change: o2o= %s psana=%s' % (o2o.attrs['_schemaVersion'],
                                                               psana.attrs['_schemaVersion'] ) ].add(o2o.name)
        tester.assertEqual(only_o2o_attr, set([]), msg="%s\nthere should be no dataset attr only in o2o" % attrDebugMsg)
        tester.assertEqual(only_psana_attr, set([]), msg="%s\nthere should be no dataset attr onlyin psana" % attrDebugMsg)

        # since o2o-translate may split epics pv's into several sources, we have to 
        # handle this separately
        if o2o.name.endswith('Epics::EpicsPv'):
            compareEpicsPvGroups(tester,o2o,psana,diffs,cmpDsetValues,recurse)
            return

        # make sure names of subgroups, datasets, links, etc are the same:
        tester.assertEqual(only_o2o_keys,set([]), 
                           msg="o2o should have no non-shared keys %s" % keyDebugMsg)

        tester.assertEqual(only_psana_keys,set([]), 
                           msg="psana should have no non-shared keys %s" % keyDebugMsg)
        o2o_notLinks = [ky for ky in o2o.keys() if type(o2o.get(ky,getlink=True)) != h5py.SoftLink]
        o2o_links = [ky for ky in o2o.keys() if type(o2o.get(ky,getlink=True)) == h5py.SoftLink]
        psana_notLinks = [ky for ky in psana.keys() if type(psana.get(ky,getlink=True)) != h5py.SoftLink]
        psana_links = [ky for ky in psana.keys() if type(psana.get(ky,getlink=True)) == h5py.SoftLink]
        o2o_notLinks.sort()
        o2o_links.sort()
        psana_notLinks.sort()
        psana_links.sort()
        tester.assertEqual(o2o_notLinks,psana_notLinks, 
                           msg="o2o and psana nonlinks differ %s" % o2o.name)
        tester.assertEqual(o2o_links,psana_links, 
                           msg="o2o and psana links differ %s" % o2o.name)
        
        # recursively check all non link items.
        common_notLinks = list(set(o2o_notLinks).intersection(set(psana_notLinks)))
        for name in common_notLinks:
            if recurse:
                compareTranslation(tester,o2o[name],psana[name],diffs,cmpDsetValues=cmpDsetValues, recurse=recurse)

    else:
        # this should be a dataset
        tester.assertEqual(type(o2o), h5py._hl.dataset.Dataset)
        tester.assertEqual(type(psana), h5py._hl.dataset.Dataset)
        
# currently we are not trying to get the same chunk size, shuffle and compression as o2o-translate
# uncomment to compare these things:
#
#        print "chunk= %s,%s  shuffle= %s,%s  compr= %s,%s  comp_opts= %s,%s  %s" % \
#            (o2o.chunks,psana.chunks, o2o.shuffle,psana.shuffle, o2o.compression, psana.compression,
#             o2o.compression_opts, psana.compression_opts, o2o.name)
#        if not o2o.name.startswith('/Configure:0000/Epics::EpicsPv/EpicsArch.0:NoDevice.0') and not \
#           o2o.name.startswith('/Configure:0000/Run:0000/CalibCycle:0000/Epics::EpicsPv/EpicsArch.0'):
#            tester.assertEqual(o2o.chunks, psana.chunks, "chunk differs in %s" % o2o.name)
#            tester.assertEqual(o2o.shuffle, psana.shuffle, "shuffle differs in %s" % o2o.name)
#            tester.assertEqual(o2o.compression,psana.compression, "compression differs in %s" % o2o.name)
        if cmpDsetValues:
            if o2o.name.find('Epics::EpicsPv')>=0:
                compareEpicsDataset(tester,o2o,psana,diffs)
            else:
                compareNonEpicsDataset(tester,o2o,psana,diffs)
                                
def compareEpicsPvGroups(tester,o2o,psana,diffs,cmpDsetValues,recurse):
    common_keys, only_o2o_keys, only_psana_keys, keyDebugMsg = groupKeys(o2o, psana, "epics pv group subgroups")

    # we are comparing a path like /Configure:0000/Epics::EpicsPv/
    # For psana, there sould only be /Configure:0000/Epics::EpicsPv/EppicsArch.0:NoDevice.0
    # o2o should have this as well, but it may also have a other sources such as AmoVMI.0:Opal1000.0
    # Whatever psana has, o2o should have as well (as long as there is at least one pv from 
    # EpicsArch.0:NoDevice.0.
    tester.assertEqual(only_psana_keys,set([]), 
                       msg="psana should have no non-shared keys %s" % keyDebugMsg)
    
    for common in common_keys:
        o2o_common = o2o[common]
        psana_common = psana[common]
        attr_diffs, only_o2o_attr, only_psana_attr, attrDebugMsg = attrDiffs(o2o, psana)
        tester.assertEqual(attr_diffs,set([]), 
                           "all epics src group attributes should be the same %s" % attrDebugMsg)
        
        tester.assertEqual(only_o2o_attr, set([]), "there should be no dataset attr only in o2o %s" % attrDebugMsg)
        # report diffs
        for ky in only_o2o_attr:
            diffs['attr only in o2o2: %s o2o=%s' % (ky, o2o.attrs[ky])].add(o2o.name)

        tester.assertEqual(only_psana_attr, set([]), 
                           msg="psana should not have any non-shared attributes %s" % attrDebugMsg)

    o2o_epicsPvLinks = []
    o2o_epicsPvs = []
    psana_epicsPvLinks = []
    psana_epicsPvs = []
    
    for pvLinks, pvs, h5obj in zip([o2o_epicsPvLinks, psana_epicsPvLinks],
                                   [o2o_epicsPvs, psana_epicsPvs],
                                   [o2o, psana]):
        for epicsSrcKey in h5obj:
            epicsSrc = h5obj[epicsSrcKey]
            for ky in epicsSrc.keys():
                if type(epicsSrc.get(ky,getlink=True)) == h5py.SoftLink:
                    pvLinks.append(epicsSrc[ky])
                else:
                    pvs.append(epicsSrc[ky])
                    
    o2oLinkNames=[x.name.split('/')[-1] for x in o2o_epicsPvLinks]
    psanaLinkNames=[x.name.split('/')[-1] for x in psana_epicsPvLinks]
    o2oLinkNames.sort()
    psanaLinkNames.sort()
    tester.assertEqual(len(o2oLinkNames), len(set(o2oLinkNames)), msg="duplicate o2o links")
    tester.assertEqual(len(psanaLinkNames), len(set(psanaLinkNames)), msg="duplicate psana links")
    commonLinks, o2oOnlyLinks, psanaOnlyLinks = partSets(o2oLinkNames, psanaLinkNames)
    if o2oOnlyLinks != set([]):
        diffs['o2o_only_epics_links'].add('  \n'.join(list(o2oOnlyLinks)))
    if psanaOnlyLinks != set([]):
        diffs['psana_only_epics_links'].add('  \n'.join(list(psanaOnlyLinks)))

    o2oNames=[x.name.split('/')[-1] for x in o2o_epicsPvs]
    psanaNames=[x.name.split('/')[-1] for x in psana_epicsPvs]
    o2oNames.sort()
    psanaNames.sort()
    o2oDupNames = [nm for idx,nm in enumerate(o2oNames[1:]) if nm == o2oNames[idx]]
    if len(o2oDupNames)>0:
        diffs['o2o has duplicate epics pv names'].add(' '.join(o2oDupNames))
    tester.assertEqual(len(psanaNames), len(set(psanaNames)), msg="duplicate psana pv names")
    commonNames, o2oOnlyNames, psanaOnlyNames = partSets(o2oNames, psanaNames)
    tester.assertEqual(o2oOnlyNames,set([]),msg="%s: o2o has epics names psana does not:\n  %s" % (o2o.name,'  \n'.join(list(o2oOnlyLinks))))
    tester.assertEqual(psanaOnlyNames,set([]),msg="%s: psana has epics names o2o does not:\n  %s" % (o2o.name,'  \n'.join(list(psanaOnlyLinks))))

    # compare all of the common epics datasets
    # index pv group objects by pvName (they may come from different sources in psana or o2o)
    o2oPvName2h5Obj=dict([(x.name.split('/')[-1],x) for x in o2o_epicsPvs])
    psanaPvName2h5Obj=dict([(x.name.split('/')[-1],x) for x in psana_epicsPvs])
    for pvName in commonNames:
        o2oEpicsPvGroup = o2oPvName2h5Obj[pvName]
        psanaEpicsGroup = psanaPvName2h5Obj[pvName]
        compareTranslation(tester, o2oEpicsPvGroup, psanaEpicsGroup, diffs, cmpDsetValues, recurse)

def compareEpicsDataset(tester,o2o,psana,diffs):
    '''o2o-translate can store epics from several pv's into the same pv name.  
    most of the time it does not, in which case this function simple compares the
    entire o2o-translate dataset against the psana one.
     
    When the shape's are different, this function tests for one pvId in the psana dataset.
    It then extracts rows with the same pvId from o2o-translate and compares them.
    
    Uses tester to assert that datasets, or rows with same pvId, are the same.  Does
    not look at o2o rows with pvIds's not equal to the psana pvId.

    However another complication is that the data can report several pvid's for the same
    epics data variable, and 
    When the dataset ends with /data, this function will also compare the /time dataset
    so that it can use the same matching rows found with the /data dataset.  When passed
    the time dataset, it skips it.

    Presently reads entire datasets.
    '''
    if o2o.name.endswith('/time'):
        return
    o2oEventTime = getDistinctTimes(o2o.parent['time'])
    psanaEventTime = getDistinctTimes(psana.parent['time'])

    o2oDistinctEventTimes = [(s,n) for s,n,ks in o2oEventTime]
    psanaDistinctEventTimes = [(s,n) for s,n,ks in psanaEventTime]
    dbgMsg = "##### epics event times: %s ###\n" % o2o.name
    dbgMsg += "o2o:\n%s\npsana:\n%s\n" % ('\n'.join(map(str,o2oDistinctEventTimes)),
                                          '\n'.join(map(str,psanaDistinctEventTimes)))
    if o2oDistinctEventTimes !=  psanaDistinctEventTimes:
        diffs["epics distinct event times differ"].add(o2o.name)

    ## make sure that if the event times are duplicated, that the data is duplicated.
    ## It is Ok for pvid to differ, do not compare that.
    ## the enum strs is complicated to compare
    ## first compare o2o, then psana
    for distinctTimes,loc,ds in zip([o2oEventTime,psanaEventTime],['o2o','psana'],[o2o,psana]):
        for s,n,ks in distinctTimes:
            if len(ks)>1:
                data = ds[ks]
                for nm in ds.dtype.names:
                    if nm.lower() == 'pvid':
                        continue
                    if nm.lower() == 'stamp':
                        secs = data[nm]['secPastEpoch']
                        nsecs = data[nm]['nsec']
                        secEqual = (secs == secs[0]).all()
                        nanoEqual = (nsecs == nsecs[0]).all()
                        allEqual = secEqual and nanoEqual
                    if nm == 'strs':
                        print "not checking strs for %s: %r" % (loc,data[nm])
                        continue
                    allEqual = (data[nm][0]==data[nm]).all()
                    if not allEqual:
                        try:
                            allEqual = all([ math.isnan(x) for x in data[nm] ])
                        except:
                            pass
                    tester.assertTrue(allEqual,msg='the duplicate data in epics pv %s from %s with the same timestamp (sec=%s nano=%s) and for field name=%s do not have the same values: %r' % \
                                      (ds.name, loc, s,n,nm,data[nm]))

    # now go through and compare epics. For each of the common times, take the first 
    # entry.
    commonTimes = list(set(o2oDistinctEventTimes).intersection(set(psanaDistinctEventTimes)))
    commonTimes.sort()
    for s,n in commonTimes:
        o2oIdx = [x[2] for x in o2oEventTime if x[0]==s and x[1]==n][0][0]
        psanaIdx = [x[2] for x in psanaEventTime if x[0]==s and x[1]==n][0][0]
        for nm in o2o.dtype.names:
            if nm.lower()=='pvid':
                continue
            elif nm.lower()=='stamp':
                for fld in ['secPastEpoch','nsec']:
                    o2oVal = o2o['stamp'][fld][o2oIdx]
                    psanaVal = psana['stamp'][fld][psanaIdx]
                    tester.assertEqual(o2oVal, psanaVal,msg=('epics %s in stamp not equal: %s o2oIdx=%d psanaIdx=%d' % \
                                   (fld,o2o.name, o2oIdx, psanaIdx)))
                continue
            elif nm.lower()=='strs':
                print "not checking strs"
                continue
            o2oVal = o2o[nm,o2oIdx]
            psanaVal = psana[nm,psanaIdx]
            equalFld = o2oVal == psanaVal
            if not equalFld and o2o.dtype[nm].name in ['float64','float32']:
                equalFld = math.isnan(o2oVal) and math.isnan(psanaVal)
            tester.assertTrue(equalFld,msg=("%s field=%s o2oIdx=%d psanaIdx=%d values differ: %r vs. %r" % \
                                            (o2o.name, nm, o2oIdx, psanaIdx, o2oVal, psanaVal)))
    # we could also check that we have the same epicsTimeStamp values between the two datasets
    
def compareNonEpicsDataset(tester,o2o,psana, diffs):
    '''tests if two non epics datasets are the same.  
    Records non-failing differences along the way.
    ARGS:
    tester - an instance of unittest.TestCase
    o2o    - dataset produced by o2o-translate
    psana  - dataset produced by psana Translator.H5Output module
    diffs  - optional dictionary where non failing differences are reported.  Keys are
             strings and values are dataset or object names where the differences were found.
    '''
    ########## helper functions ###############
    def recordFldDiffs(o2o,psana,diffs):
        if not (o2o.dtype and o2o.dtype.names):
            return
        commonFlds, o2oFlds, psanaFlds = partSets(o2o.dtype.names, psana.dtype.names)
        # compare the type of the common fields
        fldTypeDiffs = [(fld,o2o.dtype[fld],psana.dtype[fld]) for fld in commonFlds \
                        if o2o.dtype[fld] != psana.dtype[fld] ]
        if len(fldTypeDiffs)>0:
            fldDiffKey = "common flds have different types: "
            for fld,o2oType,psanaType in fldTypeDiffs:
                fldDiffKey += "%s: %s vs %s," % (fld,o2oType,psanaType)
            diffs[fldDiffKey].add(o2o.name)

        # report of differences in commond field order
        if o2o.dtype.names != psana.dtype.names:
            fldDiffKey = 'common flds orders differ:\n  o2o: %s\npsana: %s' % \
                         (' '.join(o2o.dtype.names), ' '.join(psana.dtype.names))
            diffs[fldDiffKey].add(o2o.name)

        # report on differences in field names
        if len(o2oFlds)>0 or len(psanaFlds)>0:
            fldDiffKey = "flds: %d vs %d, o2o-only: %s   psana-only: %s"
            fldDiffKey %= (len(o2o.dtype.names), len(psana.dtype.names), \
                           ' '.join(list(o2oFlds)), \
                           ' '.join(list(psanaFlds)))
            diffs[fldDiffKey].add(o2o.name)

    def compareDataValues(tester,o2o,psana,diffs):
        ##### helper function:
        def cmpDatasetFields(fldA, fldB):
            # compares two dataset fields.  Recursively descends into 
            # compound types.  Deals with Nan.  Returns a 2-list:
            # 
            #  [same, failurePath] where
            # 
            #  same:  a bool - True if the fields are the same
            #  failurePath   - a list of fieldnames that go to where the failure was
            if fldA.dtype.names:
                for nm in fldA.dtype.names:
                    subRes = cmpDatasetFields(fldA[nm], fldB[nm])
                    if subRes[0]:
                        return subRes
                    subRes[1].insert(0,nm)
                    return subRes
            else:
                equalCompare = fldA == fldB
                if not isinstance(equalCompare,bool):
                    equalCompare = equalCompare.all()
                if not equalCompare:
                    if fldA.dtype.base.name in ['float32', 'float64']:
                        noNanA = [x for x in fldA.flatten() if not math.isnan(x)]
                        noNanB = [x for x in fldB.flatten() if not math.isnan(x)]
                        equalCompare = all([x==y for x,y in zip(noNanA,noNanB)])
                if equalCompare:
                    return [True,[]]
                return [False,[]]
                
        #### end helper, start code for compareDataValues
        if o2o.name.split('/')[1] == 'time':
            return

        o2oIdx=None
        psanaIdx=None
        if 'time' in o2o.parent.keys():
            o2oIdx,psanaIdx = compareTime(tester,o2o.parent['time'],
                                          psana.parent['time'],diffs)
        if o2o.dtype and o2o.dtype.names:
            colnames,jnkA,jnkB = partSets(o2o.dtype.names,psana.dtype.names)
            colnames=list(colnames)
        else:
            colnames = [ Ellipsis ]

        for fld in colnames:
            o2oCol = None
            psanaCol = None
            try:
                o2oCol = o2o[fld]
            except:
                diffs["failed to read o2o column %s" % fld].add(o2o.name)

            try:
                psanaCol = psana[fld]
            except:
                diffs["failed to read psana column %s" % fld].add(psana.name)
 
            if (o2oCol is None) or (psanaCol is None):
                tester.assertEqual(o2oCol,psanaCol,
                                   msg="if one of o2o[%s] or psana[%s] is None, both should be. %s" % (fld,fld,o2o.name))
                # if we can't read one column, we can't read any.  These are probably
                # both NULL dataspace datasets, just return.
                return
            if o2oIdx:
                o2oCol = o2oCol[o2oIdx]
            if psanaIdx:
                psanaCol = psanaCol[psanaIdx]
            if fld == 'fifoEvents':
                assert len(o2oCol)==len(psanaCol), "fifoEvents not the same len after compareTimes: %s" % o2o.name
                for ii in range(len(o2oCol)):
                    o2oFifoEvent = o2oCol[ii]
                    psanaFifoEvent = psanaCol[ii]
                    timeStampHighEqual = (o2oFifoEvent['timestampHigh']==psanaFifoEvent['timestampHigh']).all()
                    timeStampLowEqual = (o2oFifoEvent['timestampLow']==psanaFifoEvent['timestampLow']).all()
                    eventCodeEqual = (o2oFifoEvent['eventCode']==psanaFifoEvent['eventCode']).all()
                    tester.assertTrue(timeStampHighEqual,msg=("timestamp high not equal for ii=%d, %s" % (ii,o2o.name)))
                    tester.assertTrue(timeStampLowEqual,msg=("timestamp low not equal for ii=%d, %s" % (ii,o2o.name)))
                    tester.assertTrue(eventCodeEqual,msg=("timestamp eventCode not equal for ii=%d, %s" % (ii,o2o.name)))
                    if not timeStampHighEqual or not timeStampLowEqual or not eventCodeEqual:
                        return
                return
            else:
                checkFldRes = cmpDatasetFields(o2oCol,psanaCol)
            if not checkFldRes[0]:
                print "data does not agree for %s" % o2o.name
                print "  fld is %s-%s" % (fld, '-'.join(checkFldRes[1]))
                print "*o2o*   %r" % o2oCol
                print "*psana* %r" % psanaCol
            tester.assertEqual(checkFldRes[0],True,o2o.name)

    ######## end helper functions
    # check that there are no attributes in the datasets
    tester.assertEqual(o2o.attrs.keys(),[])
    tester.assertEqual(psana.attrs.keys(),[])
    if o2o.shape != psana.shape:
        diffs['o2o.shape = %r != psana.shape = %r' % (o2o.shape, psana.shape)].add(o2o.name)
    recordFldDiffs(o2o,psana,diffs)    
    compareDataValues(tester,o2o,psana,diffs)

usage='''compareTranslation.py o2o.h5 psana.h5 [--nostop]
Pass the output of o2o-translate, and then of the psana Translator.H5Output module.
will stop if it finds a big problem between the two, report on differences otherwise.
optional --nostop means don't stop on an error, just print it
'''

def driver(args):
    if len(args) < 3:
        print usage
        return 0
    o2o_h5_fname = args[1]
    psana_h5_fname = args[2]
    tester = MockTesterStop()
    if len(args)==4:
        if args[3]=='--nostop':
            tester = MockTesterNoStop()
        else:
            print "**error, last arg: %s not understood" % args[3]
            print usage
            return -1
    assert os.path.exists(o2o_h5_fname)
    assert os.path.exists(psana_h5_fname)
    o2o_h5 = h5py.File(o2o_h5_fname,'r')
    psana_h5 = h5py.File(psana_h5_fname,'r')
    diffs = collections.defaultdict(set)
    compareTranslation(tester,o2o_h5, psana_h5, diffs)
    for ky,vals in diffs.iteritems():
        print ky
        print "  at %d locations" % len(vals)
        print "  %s" % vals.pop()
        if len(vals)>0:
            print "  %s" % vals.pop()
    return 0

if __name__ == '__main__':
    sys.exit(driver(sys.argv))

