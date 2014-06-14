#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$ 
#
# Description:
#  Module Hdf5DdlTranslator...
#
#------------------------------------------------------------------------

"""DDL parser which generates code for the Translator package.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id$ 

@author $Author$ 
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$" 
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import collections
import os
import copy

#--------------------------------
#  Imports of other modules --
#--------------------------------
import jinja2 as ji
from psddl.TemplateLoader import TemplateLoader

def getAliasAndGroupingTerm(s,endStrings):
  '''used to group together similar xtc type names
  IN: s           - string (C++ type name for xtc data) to produce alias and grouping term for
      endStrings  - list of possible grouping terms at the end of the string
  Example:
    getAliasAndGroupingTerm('CsPadData',['Config','Data'])  returns
    'CsPad', 'Data'
    getAliasAndGroupingTerm('CsPadUnknownEnd',['Config','Data'])  returns
    'CsPadUnknownEnd', ''
  '''
  for endstr in endStrings:
    if s.endswith(endstr) and len(s)>len(endstr):
      return s[0:-len(endstr)],endstr
  return s,''

def appendUnrolledAttrs(curType, attrList):
  '''Takes a package type (from the Ddl parser) and recursively goes through
  attributes until hitting basic types. Adds attributes to attrList.

  Example:
  
  import psddl.XmlReader
  EpicsPvTimeDouble = psddl.XmlReader(['psddldata/data/epics.ddl.xml'],inc_dir='.').read().packages()[0].types()[-3]
  attrList = []
  appendUnrolledAttrs(EpicsPvTimeDouble,attrList)

  Then attrList will contain one entry for each basic type in EpicsPvTimeDouble.  Each entry
  is a dictionary with 

    'offset'   the basic attribute offset within EpicsPvTimeDouble
    'accessor' list of accessors used to drill down to this basic attribute.  Start with the last 
               attribute in the list, work down two the first.
    'attr'     the basic attribute, this is either an attribute of EpicsPvTimeDouble or an attribute that
               was found by drilling down into the non-basic attributes of EpicsPvTimeDouble.

  If one takes a look at attrList for this example, one will see:

  [(x['offset'],x['accessor'],x['attr'].name) for x in attrList]

  [( 6, ['pad0'],                       'pad0'),
   ( 8, ['status()', 'dbr()'],          '_status'),
   (10, ['severity()', 'dbr()'],        '_severity'),
   (12, ['sec()', 'stamp()', 'dbr()'],  '_secPastEpoch'),
   (16, ['nsec()', 'stamp()', 'dbr()'], '_nsec'),
   (20, ['RISC_pad', 'dbr()'],          'RISC_pad'),
   (24, ['data()'],                     '_data'),
   ( 0, ['pvId()'],                     '_iPvId'),
   ( 2, ['dbrType()'],                  '_iDbrType'),
   ( 4, ['numElements()'],              '_iNumElements')]
  '''
  for attr in curType.attributes():
    assert attr.offset.isconst(), "not a constant offset: %s" % attr.name
    attrType = attr.type
    if attrType.basic:
      accessor = attr.name
      if attr.accessor:
        accessor = attr.accessor.name + '()'
      attrList.append({'offset':int(attr.offset.value), 
                       'attr':attr,
                       'accessor':[accessor]})
    else:
      startOffset = int(attr.offset.value)
      attrListCompound = []
      appendUnrolledAttrs(attrType, attrListCompound)
      for fld in attrListCompound:
        curAccessor = [ x for x in fld['accessor'] ]
        curAccessor.append(attr.accessor.name + '()')
        fld['offset'] = startOffset + fld['offset']
        fld['accessor'] = curAccessor
        attrList.append(fld)
  if curType.base:
      appendUnrolledAttrs(curType.base,attrList)

def rollUpStamp(pvType,attrList):
  '''If attrList has secPastEpoch and Nsec, remove them and replace with the stamp 
  attribute.  Roll them back up for backward compatibility with old translator.
  '''
  hasSecPastEpoch = False
  hasNsec = False
  offset = -1
  attrListStamp = []
  for attr in attrList:
    if attr['attr'].name == '_secPastEpoch':
      assert attr['accessor'] == ['sec()', 'stamp()', 'dbr()'], "unexpected accessor for secPastEpoch"
      hasSecPastEpoch = True
      offset = attr['offset']
      dbrType = ([x for x in pvType.attributes() if x.name == '_dbr'][0]).type
      try:
        stampAttribute = [x for x in dbrType.attributes() if x.name == '_stamp'][0]
        attrListStamp.append({'offset':offset,'attr':stampAttribute,'accessor':['stamp()','dbr()']})
      except:
        import pdb
        pdb.set_trace()
      continue
    elif attr['attr'].name == '_nsec':
      hasNsec = True
      continue
    attrListStamp.append(attr)  
  assert (hasSecPastEpoch and hasNsec) or ((not hasSecPastEpoch) and (not hasNsec)), \
    "unrolled should have both secsPastEpoch and nsec, or neither"
  return attrListStamp

def getAttrsAndValueForCodeGen(attrList, pvname, specialTypes):    
  '''After an epics pv type has been unrolled (and perhaps had the stamp fixed up)
  The attrList, sorted by offset in the type (which gives the order that we 
  want the attributes to appear in the hdf5 type) is passed to this function.
  This function returns elements for the code generation of the hdf5 types to 
  hold the epics pvs.
  IN:
  attrList - a list as returned by appendUnrolledAttrs and rollUpStamp, sorted by offset
             (or how the fields should appear in the h5types)
  pvname   - the epics pv type name
  specialTypes - some of the fields in the epics compound types will not be basic NATIVE types,
                 they may be strings of a certain size, or an array of strings, or the stamp.
                 These types will be passed in to the functions that create the epics types.
                 specialTypes is a dictionary of these types. keys are names for the fields, 
                 values are the variables that will hold these more complex h5 types.
  Output will be
  attrsForHeader  a list of dictionaries, each will have an entry like:
      {'accessor': 'dbr().severity()',
       'array_print_info': '',
       'assignment': 'normal',
       'basetype': 'int16_t',
       'h5name': 'severity',
       'h5type': 'H5T_NATIVE_INT16',
       'name': 'severity',
       'strncpy_max': None},
         or
      {'accessor': 'dbr().stamp()',
       'array_print_info': '',
       'assignment': 'normal',
       'basetype': 'epicsTimeStamp',
       'h5name': 'stamp',
       'h5type': 'stampType',  # assumming 'stamp:'stampType' was in the specialTypes dict
       'name': 'stamp'}
 
   value_basetype           - base type for the value of the pv (for instance 'double')
   value_array_print_info   - if an array, what to print after the variable, like '[10]'
   value_assignment         - this can be 'normal', 'strncpy' or 'enumstr', how to assign the variable
   value_strncpy_max        - if strncpy, the max bytes to copy
  value_h5type              - the h5type to use
  '''
  ### helper functions
  def type2h5type(tp):
    if tp.lower().endswith('_t'):
      return tp[0:-2]
    return tp

  def attr2h5FldName(name):
    fldname = name
    if name.lower().startswith('i'):
      fldname = name[1].lower() + name[2:]
    if fldname == 'sPvName':
      return 'pvname'
    return fldname

  ### end helper functions, start code
  attrsToWrite = [x for x in attrList if x['attr'].name.lower().find('pad')<0]
  attrsForHeader = []
  for uattr in attrsToWrite:
    attr = uattr['attr']
    name = attr.name
    if name[0]=='_':
      name = name[1:]
    basetype = attr.type.name
    if name.lower().endswith('pvname'):
      h5type = specialTypes['pvname']
    else:
      h5type = specialTypes.get(name, 'H5T_NATIVE_' + type2h5type(basetype.upper()))
    h5name = attr2h5FldName(name)
    assignment = 'normal'
    strncpy_max = None
    array_print_info = ''
    if name == 'data':  # the data field gets mapped to a value member
      value_basetype = basetype
      if pvname.endswith('String'):
        value_h5type = specialTypes['string']
      else:
        value_h5type = 'H5T_NATIVE_' + type2h5type(basetype).upper()
      value_assignment = 'normal'
      value_strncpy_max = None
      value_array_print_info = ''
      if attr.shape and len(attr.shape.dims) == 2:
        value_array_print_info = '[Psana::Epics::%s]' % attr.shape.dims[1]
        assert value_basetype == 'char', "2d value is not of char?"
        value_strncpy_max = attr.shape.dims[1]
        value_assignment = 'strncpy'
      continue
      
    if attr.shape:
      assert attr.shape.isfixed(), "non value attribute is not a fixed size array"
      newdims = ['Psana::Epics::' + x for x in attr.shape.dims]
      array_print_info = '[' + ']['.join(newdims) + ']'
      assert basetype == 'char', 'array value is not of char?'
      if len(attr.shape.dims) == 1:
        assignment = 'strncpy'
        strncpy_max = attr.shape.dims[0]
      elif len(attr.shape.dims) == 2:
        assert name == 'strs', "2d attribute that is not strs"
        assignment = 'enumstr'

    accessor = list(uattr['accessor'])
    accessor.reverse()
    accessor='.'.join(accessor)
    attrsForHeader.append({'basetype':basetype, 'name':name, 
                           'array_print_info':array_print_info,
                           'accessor':accessor,
                           'assignment':assignment,
                           'h5type': h5type,
                           'h5name': h5name,
                           'strncpy_max':strncpy_max})
  return attrsForHeader, value_basetype, value_array_print_info, value_assignment, value_strncpy_max, value_h5type

#---------------------
#  Class definition --
#---------------------
class DdlHdf5Translator ( object ) :

    @staticmethod
    def backendOptions():
        """ Returns the list of options supported by this backend, returned value is 
        either None or a list of triplets (name, type, description)"""
        return [
            ('package_dir', 'PATH', "package directory for output files"),
            ]

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, backend_options, appBaseArg ):
        """Constructor
        
            @param incname  include file name
        """
        assert 'package_dir' in backend_options, "backend options must include the " + \
               " Translator package directory. For example: package_dir:Translator"
            
        self.log = appBaseArg
        self.packageDir = backend_options['package_dir']
        self.jiEnv = ji.Environment(loader=TemplateLoader(package='Translator',
                                                          templateSubDir='templates'), 
                                    trim_blocks=True,
                                    line_statement_prefix='$',
                                    line_comment_prefix='$$')
    #-------------------
    #  Public methods --
    #-------------------

    def parseTree ( self, model ) :
        base_headers = set()
        for package in model.packages():
          for type in package.types():
            ddlfile = type.location
            basename = os.path.basename(ddlfile)
            baseheader = basename + '.h'
            if baseheader == 'xtc.ddl.h':
              continue
            base_headers.add(baseheader)
        base_headers = list(base_headers)
        base_headers.sort()
        # typeAliasMap: keys will be aliases used to refer to a collection of
        #   types, such as CsPad.  Values will be all the actual C++ types the
        #   alias refers to, such as [CsPadDataV1, CsPadConfigV1]
        # The aliases are obtained by 'stripping the end' from the *XtcTypeName*
        # in the Xml.  These names often end with things like Config or Data.
        # We form the alias by grouping Xtc TypeNames that end with the the 
        # same grouping terms:
        groupingTerms = ['Config','Data','Element','frame','Frame','config']
        typeAliasMap = collections.defaultdict(set)
        namespaces = set()
        psanaTypes = dict()
        # loop over packages in the model
        for package in model.packages():
          namespace = package.name
          namespaces.add(namespace)
          for packageType in package.types():
            if packageType.type_id:   # probably a real xtc ps type
              typename = packageType.name
              version = packageType.version
              xtcTypeId = packageType.type_id
              xtcType = xtcTypeId.replace("Id_","")
              typeAlias,groupingTerm = getAliasAndGroupingTerm(xtcType,groupingTerms)
              self.log.trace("package=%s typename=%s xtctype=%s alias=%s groupingterm=%s" % \
                             (namespace, typename, xtcType, typeAlias, groupingTerm))
              psanaTypeName = "Psana::" + namespace + '::' + typename

              # The hdf writing function for PNCCD::FullFrameV1 is presently disabled
              # so we will skip it:
              if psanaTypeName == 'Psana::PNCCD::FullFrameV1':
                continue

              isConfig = 'config-type' in packageType.tags
              psanaTypes[psanaTypeName] = isConfig
              typeAliasMap[typeAlias].add(psanaTypeName)
        
        # fix up things.  The above logic may produce more aliases than we like.
        typeAliasMap['AcqWaveform'].update(typeAliasMap['Acq'])
        del typeAliasMap['Acq']


        # If we want to combine PrincetonInfo and Princeton into the same alias, uncomment this
        # typeAliasMap['Princeton'].update(typeAliasMap['PrincetonInfo'])
        # del typeAliasMap['PrincetonInfo']

        aliasesOrderedForTemplates = []
        aliases = typeAliasMap.keys()
        aliases.sort()
        for alias in aliases:
            typeList = list(typeAliasMap[alias])
            typeList.sort()
            aliasesOrderedForTemplates.append({'alias':alias,'typelist':typeList})
        
        # make all the ndarray types we will translate
        # The ndarray's translated are described in default_psana.cfg, update that file if this
        # list is changed.
        intTypes = ['%sint%d_t' % (signed,bitsize) for signed in ['','u'] for bitsize in [8,16,32,64]]
        elemenTypes = intTypes + ['float','double']
        constTypes = ['const '+tp for tp in elemenTypes]
        elemenTypes.extend(constTypes)
        elemDimPairs = [(elem,ndim) for elem in elemenTypes for ndim in [1,2,3,4]]
        ndarrays = ['ndarray<%s,%d>' % (elem,ndim) for elem,ndim in elemDimPairs]
        ndarrayAlias = 'ndarray_types'

        # below we insert ndarray before string. This order matters,
        # some generated code may insert comments before the ndarray to distinguish these types from
        # psana types
        aliasesOrderedForTemplates.append({'alias':ndarrayAlias, 'typelist':ndarrays})
        aliasesOrderedForTemplates.append({'alias':'std_string', 'typelist':['std::string']})
        self.writeDefaultPsanaCfg(aliasesOrderedForTemplates, ndarrayAlias)
        self.writeTypeAliasesCpp(base_headers, aliasesOrderedForTemplates)
        self.writeHdfWriterMapCpp(base_headers, namespaces, psanaTypes, elemDimPairs)
        epicsPackage = [package for package in model.packages() if package.name.lower()=='epics']
        assert len(epicsPackage)==1, "could not find epics package, names are: %r" % ([x.name for x in model.packages()],)
        epicsPackage = epicsPackage[0]
        self.writeEpicsHdfWriterDetails(epicsPackage)

    def writeDefaultPsanaCfg(self,aliasesOrderedForTemplates, ndarrayAlias):
        tmpl = self.jiEnv.get_template('hdf5Translator.tmpl?psana_cfg_template')
        type_filter_options = ''
        for entry in aliasesOrderedForTemplates:
            alias,typeList = entry['alias'],entry['typelist']
            lns = ''
            if alias == ndarrayAlias:
              lns += "\n# user types to translate from the event store\n"
            lns += ('%s = include' % alias).ljust(30)
            lns += ' # %s' % ', '.join(typeList)
            lns += '\n'
            type_filter_options += lns
        fname = os.path.join(self.packageDir, "data", "default_psana.cfg")
        fout = file(fname,'w')
        fout.write(tmpl.render(locals()))
        
    def writeTypeAliasesCpp(self, base_headers, aliasesOrderedForTemplates):
        class Entry(object):
            def __init__(self,alias,typeList):
                self.alias=alias
                self.typeList=typeList

        type_aliases = []
        for entry in aliasesOrderedForTemplates:
            alias,typeList = entry['alias'],entry['typelist']
            type_aliases.append(Entry(alias,typeList))
        tmpl = self.jiEnv.get_template('hdf5Translator.tmpl?type_aliases_cpp')
        fname = os.path.join(self.packageDir, 'src', 'TypeAliases.cpp')
        fout = file(fname,'w')
        fout.write(tmpl.render(locals()))

    def writeHdfWriterMapCpp(self, base_headers, namespaces, psanaTypes, elemDimPairs ):
        tmpl = self.jiEnv.get_template('hdf5Translator.tmpl?hdfwritermap_cpp')
        fname = os.path.join(self.packageDir, 'src', 'HdfWriterMap.cpp')
        fout = file(fname,'w')
        psana_types = psanaTypes.keys()
        psana_types.sort()
        # fix ups
        namespaces = [ns for ns in namespaces if ns not in ['Pds']]
        fout.write(tmpl.render(locals()))

    def writeEpicsHdfWriterDetails(self,epicsPackage):
      # ----- helper functions ------
      def inUnrollSet(typename):
        if typename.startswith("EpicsPvCtrl") and not typename.endswith("Header"):
          return True
        if typename.startswith("EpicsPvTime") and not typename.endswith("Header"):
          return True
        return False

      # ----- end helper functions, begin code --------
      epics_h_tmpl = self.jiEnv.get_template('hdf5Translator.tmpl?epics_ddl_h')
      epics_cpp_tmpl = self.jiEnv.get_template('hdf5Translator.tmpl?epics_ddl_cpp')
      dispatch_cpp_tmpl = self.jiEnv.get_template('hdf5Translator.tmpl?dispatch_cpp')

      epicsPvs = []
      specialTypes = {'units':'unitsType',
                      'pvname':'pvNameType',
                      'strs':'allEnumStrsType',
                      'string':'stringType',
                      'stamp':'stampType'}

      for pvType in epicsPackage.types():
        typename = pvType.name
        if not inUnrollSet(typename):
          continue
        attrList = []
        appendUnrolledAttrs(pvType, attrList)
        attrList = rollUpStamp(pvType,attrList)
        toSort = [(x['offset'],x) for x in attrList]
        toSort.sort()
        attrList = [x[1] for x in toSort]
        attrsForCodeGen, value_basetype, \
          value_array_print_info, value_assignment, \
          value_strncpy_max, value_h5type  = getAttrsAndValueForCodeGen(attrList, typename, specialTypes)
        createTypeArgs = 'hid_t pvNameType, hid_t unitsType'
        type_create_args = ''
        if typename.find('Ctrl')>0:
          if typename.find('Enum')>0:
            type_create_args = 'hid_t %s, hid_t strsArrayType, int numberOfStrings' % (specialTypes['pvname'],)
          elif typename.find('String')>0:
            type_create_args = 'hid_t %s, hid_t %s' % (specialTypes['pvname'],
                                                       specialTypes['string'])
          else:
            type_create_args = 'hid_t %s, hid_t %s' % (specialTypes['pvname'],
                                                       specialTypes['units'])
        elif typename.find('Time')>0:
          if typename.find('String')>0:
            type_create_args = 'hid_t %s, hid_t %s' % (specialTypes['string'], specialTypes['stamp'])
          else:
            type_create_args = 'hid_t %s' % specialTypes['stamp']
            
            
        epicsPvs.append({'name':typename, 
                         'attrs':attrsForCodeGen, 
                         'value_basetype': value_basetype, 
                         'value_array_print_info':value_array_print_info,
                         'value_assignment': value_assignment,
                         'value_strncpy_max': value_strncpy_max,
                         'value_h5type': value_h5type,
                         'type_create_args': type_create_args})
      
      fname = os.path.join(self.packageDir, 'include', 'epics.ddl.h')
      fout = file(fname,'w')
      fout.write(epics_h_tmpl.render(locals()))
      fout.close()

      fname = os.path.join(self.packageDir, 'src', 'epics.ddl.cpp')
      fout = file(fname,'w')
      fout.write(epics_cpp_tmpl.render(locals()))
      fout.close()

      dbrTypes = []
      for epicsPv in epicsPvs:
        psanaName = "Psana::Epics::" + epicsPv['name']
        pvVar = epicsPv['name'].split('EpicsPv')[1]
        try:
          dbr_const = 'DBR_TIME_' + pvVar.split('Time')[1].upper()
        except IndexError:
          dbr_const = 'DBR_CTRL_' + pvVar.split('Ctrl')[1].upper()
        dbrTypes.append({'dbr_str':dbr_const, 'pv_type':epicsPv['name']})
      fname = os.path.join(self.packageDir, 'src', 'HdfWriterEpicsPvDispatch.cpp')
      fout = file(fname,'w')
      fout.write(dispatch_cpp_tmpl.render(locals()))
#
#  In case someone decides to run this module
#
usage = '''Module is not supposed to be run as a main module.
Invoke using psddlc from the psddl package. Run from a 
release directory where both the psddldata and the Translator
packages are checked out. Run as:

psddlc -I data -B package_dir:Translator -b hdf5Translator data/psddldata/*.ddl
'''

if __name__ == "__main__" :
    sys.exit ( usage )
