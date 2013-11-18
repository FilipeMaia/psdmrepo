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
#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
import jinja2 as ji
from psddl.TemplateLoader import TemplateLoader

#------------------------
# Exported definitions --
#------------------------

def getAliasAndGroupingTerm(s,endStrings):
  for endstr in endStrings:
    if s.endswith(endstr) and len(s)>len(endstr):
      return s[0:-len(endstr)],endstr
  return s,''

def appendUnrolledAttrs(curType, attrList):
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

def unrollAttr2str(uattr):
  attr = uattr['attr']
  res = '%4d %20s fixed=%s' % (uattr['offset'], 
                               attr.name, 
                               int(attr.isfixed()))
  typestr = attr.type.name
  if attr.shape:
    typestr += '[' + ','.join(attr.shape.dims) + ']'
  res += ' %20s' % typestr
  
  res += ' %s' % uattr['accessor']
  return res

def getAttrsAndValueForCodeGen(attrList, pvname, specialTypes):    
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
    elif name == 'units':
      h5type = specialTypes['units']
    elif name == 'strs':
      h5type = specialTypes['enumstrs']
    else:
      h5type = 'H5T_NATIVE_' + type2h5type(basetype.upper())
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
        self.jiEnv = ji.Environment(loader=TemplateLoader(), trim_blocks=True,
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
            baseheader = os.path.splitext(basename)[0] + '.h'
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
              isConfig = 'config-type' in packageType.tags
              psanaTypes[psanaTypeName] = isConfig
              typeAliasMap[typeAlias].add(psanaTypeName)
        
        # fix up things.  The above logic may produce more aliases than we like.
        typeAliasMap['AcqWaveform'].update(typeAliasMap['Acq'])
        del typeAliasMap['Acq']

        # If we want to combine PrincetonInfo and Princeton into the same alias, uncomment this
        # typeAliasMap['Princeton'].update(typeAliasMap['PrincetonInfo'])
        # del typeAliasMap['PrincetonInfo']

        sortedAliases = []
        aliases = typeAliasMap.keys()
        aliases.sort()
        for alias in aliases:
            typeList = list(typeAliasMap[alias])
            typeList.sort()
            sortedAliases.append({'alias':alias,'typelist':typeList})
        
        self.writeDefaultPsanaCfg(sortedAliases)
        self.writeTypeAliasesCpp(base_headers, sortedAliases)
        self.writeHdfWriterMapCpp(base_headers, namespaces, psanaTypes)
        epicsPackage = [package for package in model.packages() if package.name.lower()=='epics']
        assert len(epicsPackage)==1, "could not find epics package, names are: %r" % ([x.name for x in model.packages()],)
        epicsPackage = epicsPackage[0]
        self.writeEpicsHdfWriterDetails(epicsPackage)

    def writeDefaultPsanaCfg(self,sortedAliases):
        tmpl = self.jiEnv.get_template('hdf5Translator.tmpl?psana_cfg_template')
        type_filter_options = ''
        for entry in sortedAliases:
            alias,typeList = entry['alias'],entry['typelist']
            ln = '%s = include' % alias
            ln = ln.ljust(30)
            ln += ' # %s' % ', '.join(typeList)
            ln += '\n'
            type_filter_options += ln
        fname = os.path.join(self.packageDir, "default_psana.cfg")
        fout = file(fname,'w')
        fout.write(tmpl.render(locals()))
        
    def writeTypeAliasesCpp(self, base_headers, sortedAliases):
        class Entry(object):
            def __init__(self,alias,typeList):
                self.alias=alias
                self.typeList=typeList

        type_aliases = []
        for entry in sortedAliases:
            alias,typeList = entry['alias'],entry['typelist']
            type_aliases.append(Entry(alias,typeList))
        tmpl = self.jiEnv.get_template('hdf5Translator.tmpl?type_aliases_cpp')
        fname = os.path.join(self.packageDir, 'src', 'TypeAliases.cpp')
        fout = file(fname,'w')
        fout.write(tmpl.render(locals()))

    def writeHdfWriterMapCpp(self, base_headers, namespaces, psanaTypes):
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
                      'enumstrs':'allEnumStrsType',
                      'string':'stringType'}
      for pvType in epicsPackage.types():
        typename = pvType.name
        if not inUnrollSet(typename):
          continue
        attrList = []
        appendUnrolledAttrs(pvType, attrList)
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
            type_create_args = 'hid_t %s, hid_t %s' % (specialTypes['pvname'],
                                                       specialTypes['enumstrs'])
          elif typename.find('String')>0:
            type_create_args = 'hid_t %s, hid_t %s' % (specialTypes['pvname'],
                                                       specialTypes['string'])
          else:
            type_create_args = 'hid_t %s, hid_t %s' % (specialTypes['pvname'],
                                                       specialTypes['units'])
        elif typename.find('Time')>0 and typename.find('String')>0:
          type_create_args = 'hid_t %s' % specialTypes['string']
            
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

psddlc -I data -B package_dir:Translator -b hdf5Translator data/psddldata/*.xml
'''

if __name__ == "__main__" :
    sys.exit ( usage )
