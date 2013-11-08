import os
import sys
import collections
import math
import h5py

class MockTester(object):
    def assertTrue(self,a,msg=''):
# uncomment this to run through all datasets, see all differences
#        if not a:
#            print "assertTrue - a is false, msg=%s" % msg
#            return
        assert a,"a is not true, msg=%s" % msg

    def assertEqual(self,a,b,msg=''):
# uncomment this to run through all datasets, see all differences
#        if a != b:
#            print "**a!=b\n  a=%r\n  b=%r\n  msg=%s" % (a,b,msg)
#        return
        assert a==b,"a !=b, msg=%s, a=%r, b=%r" %(msg,a,b)
            
def compareTranslation(tester,o2o, psana, diffs, cmpDsetValues=True, recurse=True):
    '''High level function that recursively compares h5py objects from o2o-translate and
    the psana Translator.
    ARGS:
       tester          - an instance of a class that has a method:
                         assertEqual(a,b,msg)   (such as unittest.TestCase)
       o2o, psana      - the h5py objects. Generally these are both the result of 
                         calling h5py.File.
       diffs           - a defaultdictionary with a set as the default (supports add method).
       cmpDsetValues   - defaults to True, which compare all values in the datasets. 
                         set to False to just compare the group hiearchy

    Tests that the hiearchies are the same.  Defaults to test that the datasets are the same.
    There are some acceptable differences between o2o-translate and psana.  
    These kinds of differences do not cause the tester to fail.  They are added to the
    diffs dictionary - keys are string messages and values are lists of dataset names, 
    or h5 object names, where that difference occurred.
    '''
    attr_diffs, only_o2o_attr, only_psana_attr, attrDebugMsg = attrDiffs(o2o, psana)
    attrDebugMsg = o2o.name + '\n' + attrDebugMsg

    if type(o2o)!=h5py._hl.dataset.Dataset:
        common_keys, only_o2o_keys, only_psana_keys = partSets(o2o.keys(), psana.keys())
        keyDebugMsg = "%s\ncommon keys: %s\n" % (o2o.name,' '.join(common_keys))
        keyDebugMsg += "  o2o keys: %s\n" % ' '.join(only_o2o_keys)
        keyDebugMsg += "psana keys: %s\n" % ' '.join(only_psana_keys)
        
    if type(o2o)==h5py._hl.files.File:
        # make sure all common file attributes are the same except for the following:
        tester.assertEqual(attr_diffs,set([u'origin', u'UUID', u'created', u':schema:version']), 
                           msg="The only file attributes that differ are wrong: %s" % attrDebugMsg)

        # make sure the only file attributes in one or the other are the following:
        tester.assertEqual(only_o2o_attr, set(['runType','runNumber']), 
                           msg="file attributes unique to o2o are wrong: %s" % attrDebugMsg)
        tester.assertEqual(only_psana_attr, set(['instrument','jobName','expNum']), 
                           msg="file attributes unique to psana are wrong %s" % attrDebugMsg)

        # make sure the file only has one group - Configure:0000
        tester.assertEqual(common_keys,set(['Configure:0000']), 
                           msg="common group from root is wrong. Should be Configure:0000.\n %s" % keyDebugMsg)
        tester.assertEqual(only_o2o_keys,set([]), 
                           msg="o2o should have no keys that aren't common.\n%s" % keyDebugMsg)
        tester.assertEqual(only_psana_keys,set([]), 
                           msg="psana should have no keys that aren't common.\n%s" % keyDebugMsg)
        
        # recursively check the configure group
        if recurse:
            compareTranslation(tester,o2o['Configure:0000'], psana['Configure:0000'], diffs, cmpDsetValues)

    elif type(o2o)!=h5py._hl.dataset.Dataset:
        # all common group attributes, save schema are the same
        tester.assertTrue((attr_diffs == set([]) or (attr_diffs == set(['_schemaVersion']))),
                          "the only differing group attribute should be _schemaVersion %s" % attrDebugMsg)
        
        # report differences in schema version
        if attr_diffs == set(['_schemaVersion']):
            diffs['_schemaVersion change: o2o= %s psana=%s' % (o2o.attrs['_schemaVersion'],
                                                               psana.attrs['_schemaVersion'] ) ].add(o2o.name)
        # TODO: when DDL store routines add schema, turn this on:
        tester.assertEqual(only_o2o_attr, set([]), "there should be no dataset attr only in o2o %s" % attrDebugMsg)
        # report any difference for now, report diffs
        if len(only_o2o_attr)>0:
            for ky in only_o2o_attr:
                diffs['attr only in o2o2: %s o2o=%s' % (ky, o2o.attrs[ky])].add(o2o.name)

        tester.assertEqual(only_psana_attr, set([]), 
                           msg="psana should not have any non-shared attributes %s" % attrDebugMsg)

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
                compareTranslation(tester,o2o[name],psana[name],diffs,cmpDsetValues)
    else:
        # this should be a dataset
        tester.assertEqual(type(o2o), h5py._hl.dataset.Dataset)
        tester.assertEqual(type(psana), h5py._hl.dataset.Dataset)
        
# currently we are not trying to get the same chunk size, shuffle and compression as o2o-translate
# uncomment to check these things:
#
#        print "chunk= %s,%s  shuffle= %s,%s  compr= %s,%s  comp_opts= %s,%s  %s" % \
#            (o2o.chunks,psana.chunks, o2o.shuffle,psana.shuffle, o2o.compression, psana.compression,
#             o2o.compression_opts, psana.compression_opts, o2o.name)
#        tester.assertEqual(o2o.chunks, psana.chunks, "chunk differs in %s" % o2o.name)
#        tester.assertEqual(o2o.shuffle, psana.shuffle, "shuffle differs in %s" % o2o.name)
#        test.assertEqual(o2o.compression,psana.compression, "compression differs in %s" % o2o.name)
        if o2o.name.find('Epics::EpicsPv')>=0:
            if cmpDsetValues:
                checkEpicsDataset(tester,o2o,psana,diffs)
        else:
            checkNonEpicsDataset(tester,o2o,psana,diffs,cmpDsetValues)

def compareEpicsPvGroups(tester,o2o,psana,diffs,cmpDsetValues,recurse):
    common_keys, only_o2o_keys, only_psana_keys = partSets(o2o.keys(), psana.keys())
    keyDebugMsg = "%s\ncommon keys: %s\n" % (o2o.name,' '.join(common_keys))
    keyDebugMsg += "  o2o keys: %s\n" % ' '.join(only_o2o_keys)
    keyDebugMsg += "psana keys: %s\n" % ' '.join(only_psana_keys)

    # o2o may have distinct keys, such as AmoVMI.0:Opal1000.0, since it uses several sources
    # for epics variables.  But psana should not have any distinct ones (assumming there is
    # at least one pv from EpicsArch.0:NoDevice.0. in the xtc's).
    tester.assertEqual(only_psana_keys,set([]), 
                       msg="psana should have no non-shared keys %s" % keyDebugMsg)
    
    for common in common_keys:
        o2o_common = o2o[common]
        psana_common = psana[common]
        attr_diffs, only_o2o_attr, only_psana_attr, attrDebugMsg = attrDiffs(o2o, psana)
        attrDebugMsg = o2o.name + '\n' + attrDebugMsg
        tester.assertEqual(attr_diffs,set([]), 
                           "all epics src group attributes should be the same %s" % attrDebugMsg)
        
        # TODO: when DDL store routines add schema, turn this on:
        tester.assertEqual(only_o2o_attr, set([]), "there should be no dataset attr only in o2o %s" % attrDebugMsg)
        # report diffs
        if len(only_o2o_attr)>0:
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
    tester.assertEqual(o2oOnlyLinks,set([]),msg="%s: o2o has epics links psana does not:\n   %s" % (o2o.name,'  \n'.join(list(o2oOnlyLinks))))
    tester.assertEqual(psanaOnlyLinks,set([]),msg="%s: psana has epics links o2o does not:\n   %s" % (o2o.name,'  \n'.join(list(psanaOnlyLinks))))

    o2oNames=[x.name.split('/')[-1] for x in o2o_epicsPvs]
    psanaNames=[x.name.split('/')[-1] for x in psana_epicsPvs]
    o2oNames.sort()
    psanaNames.sort()
    tester.assertEqual(len(o2oNames), len(set(o2oNames)), msg="duplicate o2o pv names")
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

def partSets(A,B):
    A = set(A)
    B = set(B)
    inter = A.intersection(B)
    onlyA = A.difference(B)
    onlyB = B.difference(A)
    return inter,onlyA,onlyB

def attrDiffs(oA,oB):
    '''takes two h5py objects.  
    Return a 3-tuple:  diffs, onlyA, onlyB  
       where
    diffs - attributes that exist in both oA and oB but have different values.  
            The key names are returned, not the differing values.
    onlyA - those attribute keys only in oA.
    onlyB - sim
    '''
    Akeys = oA.attrs.keys()
    Bkeys = oB.attrs.keys()
    Aset=set(Akeys)
    Bset=set(Bkeys)
    assert len(Aset)==len(Akeys)
    assert len(Bset)==len(Bkeys)    
    common,onlyA,onlyB = partSets(Aset,Bset)
    common_diffs = []
    debugMsg = "common attributes\n"
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
    return set(common_diffs),onlyA,onlyB,debugMsg

def checkEpicsDataset(tester,o2o,psana,diffs):
    '''o2o-translate can store epics from several pv's into the same pv name.  
    most of the time it does not, in which case this function simple compares the
    entire o2o-translate dataset against the psana one.
     
    When the shape's are different, this function tests for one pvId in the psana dataset.
    It then extracts rows with the same pvId from o2o-translate and compares them.
    
    Uses tester to assert that datasets, or rows with same pvId, are the same.  Does
    not look at o2o rows with pvIds's not equal to the psana pvId.
    
    When the dataset ends with /data, this function will also check the /time dataset
    so that it can use the same matching rows found with the /data dataset.  When passed
    the time dataset, it skips it.

    Presently reads entire datasets.
    '''
    ## checkEpics helper function:
    def cmpEpicsDataSameShape(o2o,psana):
        # compares epics datasets - watches out for possible schema change with 'stamp'
        checkEqual = True
        for nm in o2o.dtype.names:
            if nm == 'stamp':
                continue
            o2onm = o2o[nm]
            psananm = psana[nm]
            if nm != 'strs':
                # for the control enum, o2o only stores the number of strings that are
                # included, while DDL based stores them all.
                nmEqual = (o2o[nm]==psana[nm]).all()
            else:
                assert len(o2onm)==1
                assert len(psananm)==1
                o2ostrs = o2onm[0]
                psanastrs = psananm[0]
                nmEqual = (o2ostrs == psanastrs[0:psana['no_str'][0]]).all()
            if not nmEqual and o2o.dtype[nm].name in ['float64','float32']:
                o2oNoNan = [x for x in o2o[nm] if not math.isnan(x)]
                psanaNoNan = [x for x in psana[nm] if not math.isnan(x)]
                nmEqual = all([x==y for x,y in zip(o2oNoNan, psanaNoNan)])
            checkEqual &= nmEqual
            if not checkEqual:
                break

        if checkEqual and 'stamp' in o2o.dtype.names and 'stamp' not in psana.dtype.names:
            checkEqual &= (o2o['stamp']['secPastEpoch']==psana['secPastEpoch']).all()
            checkEqual &= (o2o['stamp']['nsec']==psana['nsec']).all()
        elif checkEqual and 'stamp' in o2o.dtype.names and 'stamp' in psana.dtype.names:
            checkEqual &= (o2o['stamp']['secPastEpoch']==psana['stamp']['secPastEpoch']).all()
            checkEqual &= (o2o['stamp']['nsec']==psana['stamp']['nsec']).all()
        return checkEqual

    ### begin checkEpicsDataset
    if o2o.name.endswith('/time'):
        return
    o2oTime = o2o.parent['time']
    psanaTime = psana.parent['time']
    if o2o.shape == psana.shape:
        isEqual = cmpEpicsDataSameShape(o2o,psana)
        if not isEqual:
            print "  o2o: %r" % o2o[...]
            print "psana: %r" % psana[...]
        tester.assertEqual(isEqual,True,"checkEpics - same shape but different values: %s" % o2o.name)
        timeCmp = (o2oTime[...]==psanaTime[...]).all()
        if not timeCmp:
            print "  o2o-time: %r" % o2oTime[...]
            print "psana-time: %r" % psanaTime[...]
        tester.assertEqual(timeCmp,True,
                           "checkEpics - time, same shape but different values: %s" % o2oTime.name)
        return
    diffs['Epics datasets with different shapes'].add(o2o.name)
    psanaPvIds =  set(psana['pvId'])
    tester.assertEqual(len(psanaPvIds),1, 
                       "psana epics variable has more than one pvId? ids: %s, name=%s" % \
                       (psanaPvIds, o2o.name))
    psanaPvId = list(psanaPvIds)[0]
    matching = o2o['pvId'] == psanaPvId
    o2oCompare = o2o[matching]
    checkEqual = cmpEpicsDataSameShape(o2oCompare,psana)
    if not checkEqual:
        print "  o2o: %r" % o2oCompare[...]
        print "psana: %r" % psana[...]
    tester.assertEqual(checkEqual,True,
                       "checkEpics - different shapes and different values: %s" % \
                       o2o.name)
    timeCmp = (o2oTime[matching][...]==psanaTime[...]).all()
    if not timeCmp:
        print "  o2o-time: %r" % o2oTime[matching][...]
        print "psana-time: %r" % psanaTime[...]
    tester.assertEqual(timeCmp, True,
                       "checkEpics - time - different shapes and different values: %s" % \
                       o2oTime.name)


def checkNonEpicsDataset(tester,o2o,psana, diffs, cmpValues = True):
    '''tests if two non epics datasets are the same.  
    Records non-failing differences along the way.
    ARGS:
    tester - an instance of unittest.TestCase
    o2o    - dataset produced by o2o-translate
    psana  - dataset produced by psana Translator.H5Output module
    diffs  - optional dictionary where non failing differences are reported.  Keys are
             strings and values are dataset or object names where the differences were found.
    cmpValues - compare all values for common fields in datasets
    '''
    ########## helper functions ###############
    def recordFldDiffs(o2o,psana,diffs):
        if not (o2o.dtype and o2o.dtype.names):
            return
        commonFlds, o2oFlds, psanaFlds = partSets(o2o.dtype.names, psana.dtype.names)
        # check the type of the common fields
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

    def checkDataValues(tester,o2o,psana,diffs):
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
                
        #### end helper, start code for checkDataValues
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
    tester.assertEqual(o2o.shape, psana.shape,"non epics datasets shapes differ: %s" % o2o.name)
    recordFldDiffs(o2o,psana,diffs)    
    if cmpValues:
        checkDataValues(tester,o2o,psana,diffs)

usage='''compareTranslation.py o2o.h5 psana.h5
Pass the output of o2o-translate, and then of the psana Translator.H5Output module.
will stop if it finds a big problem between the two, report on differences otherwise.
'''

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print usage
        sys.exit(0)
    o2o_h5_fname = sys.argv[1]
    psana_h5_fname = sys.argv[2]
    assert os.path.exists(o2o_h5_fname)
    assert os.path.exists(psana_h5_fname)
    o2o_h5 = h5py.File(o2o_h5_fname,'r')
    psana_h5 = h5py.File(psana_h5_fname,'r')
    tester = MockTester()
    diffs = collections.defaultdict(set)
    compareTranslation(tester,o2o_h5, psana_h5, diffs)
    for ky,vals in diffs.iteritems():
        print ky
        print "  at %d locations" % len(vals)
        print "  %s" % vals.pop()
        if len(vals)>0:
            print "  %s" % vals.pop()

