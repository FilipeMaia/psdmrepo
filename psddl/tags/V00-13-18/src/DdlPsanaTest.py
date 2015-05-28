#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$ 
#
# Description:
#  Module DdlPsanaTest
#
#------------------------------------------------------------------------

"""DDL parser which generates code for the psana_test package.
It generates the psddl_dump.py module which supports the python dump 
function. psddl_dump provides a function to dump any of the psana 
objects to a string.

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
import os

#--------------------------------
#  Imports of other modules --
#--------------------------------
import jinja2 as ji
from psddl.TemplateLoader import TemplateLoader
from collections import defaultdict
import psddl

#---------------------
#  Class definition --
#---------------------
class DdlPsanaTest ( object ) :

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
    assert 'package_dir' in backend_options, "backend options must include the " + \
      " psana_test package directory. For example: package_dir:psana_test"
    self.log = appBaseArg
    self.packageDir = backend_options['package_dir']
    assert os.path.exists(self.packageDir), "package dir %s does not exist" % self.packageDir
    self.jiEnv = ji.Environment(loader=TemplateLoader(package='psana_test',
                                                      templateSubDir='templates',
                                                      ), 
                                trim_blocks=True,
                                line_statement_prefix='$',
                                line_comment_prefix='$$')
  #-------------------
  #  Public methods --
  #-------------------
  def parseTree ( self, model ) :
    psddl_dump_py_fname = os.path.join(self.packageDir, 'src', 'psddl_dump.py')
    psddl_dump_py = file(psddl_dump_py_fname,'w')

    xtcTypes = getXtcTypes(model)
    xtc_dispatch_list = makePythonXtcDispatchList(xtcTypes)

    psana_types = makePythonPsanaTypes(model)

    python_tmpl = self.jiEnv.get_template('psana_test.tmpl?psddl_dump_py')
    psddl_dump_py.write(python_tmpl.render(xtc_dispatch_list=xtc_dispatch_list,
                                           psana_types=psana_types))
    
    psddl_dump_py.close()
                        
    
def getXtcTypes(model):
  xtcTypes={}
  for package in model.packages():
    namespace = package.name
    for ptype in package.types():
      if ptype.location.find('epics')>=0:
        continue
      if ptype.type_id:
        xtcTypes[(namespace,ptype.name)]=ptype
  return xtcTypes

BASIC_TYPES =  ['int8_t', 'int16_t', 'int32_t', 'int64_t', 'uint8_t',
                'uint16_t', 'uint32_t', 'uint64_t', 'double', 'float']

def isBasicType(ptype):
  return  ptype.name in BASIC_TYPES

def makePythonXtcDispatchList(xtcTypes):
  '''Returns a list of 3-tuples. Each 3-tuple will have the entries:

    (typeid, version, printFunctionName)

    That is a typeid, version, and print function for objects of this type.
    
    There is one case where two psana types map to the same typeid, version -
    This is PNCCD.FullFrameV1 and PNCCD.FramesV1. We exclude FullFrameV1.

    The typeId and version will be found in the psana module, i.e:

        psana.Lusi.IpmFexConfigV2.TypeId
        psana.Lusi.IpmFexConfigV2.Version

    The printFunctionName follow the format:
  
       namespace_classname_to_str

    for example:   Lusi_IpmFexConfigV2_to_str
  '''
  xtc_dispatch_list = []
  keys = xtcTypes.keys()
  keys.sort()
  for key in keys:
    xtcType = xtcTypes[key]
    ns,cl = key
    if ns == 'PNCCD' and cl == 'FullFrameV1':
      continue
    fnName = ns + '_' + cl + '_to_str'
    xtc_dispatch_list.append(('psana.' + ns + '.' + cl + '.TypeId', 
                              xtcType.version, 
                              fnName))
  return xtc_dispatch_list

# Presently providing special handling for a few methods that return a basic 
# non array and take an argument. In the future we should figure this out from the
# parsed DDL. In this dictionary, the key describes 
# the method as ( namespace.classname , method name )
# the values are (category, subMethod)
#   where category = skip  means we won't dump this, not suitable for some reason
#                    0_to_method:  we will dump [0:n] values of this method, and arg 2
#                                  is the name of a method that returns n
#                    0_to_const:   dump [0:n] and arg to is n
#                    ndarray_from: we should return an ndarray of values made by
#                                  calling this method over indicies into the array.
#                                  the shape of the array to form is the same as that
#                                  returned by the method named in arg 2

basic_arg_not_array = {('Alias.SrcAlias','operator<'):('skip',None),
                       ('Alias.SrcAlias','operator=='):('skip',None),
                       ('CsPad.ConfigV','roiMask'):('0_to_method','obj.numQuads()'),
                       ('CsPad.ConfigV','numAsicsStored'):('0_to_method','obj.numQuads()'),
                       ('Encoder.DataV2','value'):('0_to_const','3'),
                       ('OceanOptics.DataV','nonlinerCorrected'):('ndarray_from','data')}

def getBasicNotArrayRule(namespace, classname, methodname):
  nc = '%s.%s' % (namespace, classname)
  for key,rule in basic_arg_not_array.iteritems():
    keyNc, keyMethod = key
    if nc.startswith(keyNc):
      if methodname == keyMethod:
        return rule
  return None

def getNdarrayDumpFunction(ptype):
  if isBasicType(ptype):
    return 'ndarray_to_str'
  return None
  
def getDumpFunction(ptype, namespace=None):
  '''Returns the name of the dump function.
  ARGS:
  ptype - a psddl type.
  namespace - ignored for basic types, but if not, used for form
               dump function name for non basic type.
  '''
  if isinstance(ptype,psddl.Enum.Enum):
    return 'enum_to_str'
  typename = ptype.name
  if isBasicType(ptype):
    if typename.endswith('_t'):
      typename = typename[0:-2]
    return '%s_to_str' % typename
  if typename == 'char':
    return 'str_to_str'
  assert namespace is not None or len(namespace)>0, "non basic type: %s but no namespace" % typename
  return "%s_%s_to_str" % (namespace, typename)

class PsanaType(object):
  # placeholder class for method information for generating dump code
  pass

def makePythonPsanaTypes(model):
  psana_types = []
  for package in model.packages():
    for ptype in package.types():
      if ptype.location.find('epics')>=0:
        continue
      if ptype.name == 'Src' and package.name in ['Partition', 'Pds']:
        continue
      if ptype.name == 'DetInfo' and package.name == 'Pds':
        continue
      xtctype = False
      if ptype.type_id:
        xtctype = True
      psana_type = categorize(ptype)
      psana_type.namespace = package.name
      psana_type.xtctype = xtctype
      psana_type.classname = ptype.name
      psana_types.append(psana_type)
  return psana_types

def getBasicAttrNoAccessor(ptype):
  '''returns list of basic attributes that have no method accessor.
  (There are no non-basic attributes without a method accessor).
  ARG:
    ptype - type as returned by DDL parser
  OUT:
    basic_attr   - a list of (attrName, dumpFunction) 
                   such as [('voltage', 'float_2_str'), 
                            ('rows','int32_2_str')]
  '''
  attributesWithNoAccessors = [attrib \
                               for attrib in ptype.attributes() \
                               if attrib.access=='public' and \
                               (not attrib.name.startswith('_')) and \
                               not (attrib.accessor and attrib.accessor.access=='public')]

  attributesBasic = [(attrib.name, attrib.type.name) \
                     for attrib in attributesWithNoAccessors \
                     if (attrib.shape is None) and (attrib.type.name in BASIC_TYPES)]

  if len(attributesBasic) != len(attributesWithNoAccessors):
    msg =  "**WARNING** DdlPsanaTest: %s" % ptype
    msg += " has attributes with no accessor that are not basic types."
    msg += " no code generated to dump values. Attributes are:"
    nonBasicNames = [attr.name for attr in attributesWithNoAccessors \
                     if (attrib.shape is not None) or (not isBasicType(attrib.type.name))]
    msg += ' '.join(nonBasicNames)
    print msg

  basic_attr = [(attrname, getDumpFunction(attrtype)) for attrname, attrtype in attributesBasic]
  return basic_attr

def getMethodReturnTypeNamespace(method):
  methodReturnTypeNamespace = ''
  try:
    methodReturnTypeNamespace = method.type.package.name
  except AttributeError:
    pass
  return methodReturnTypeNamespace

def getSizeExpr(method):
  attr = method.attribute
  if not attr:
    return None
  if not attr.shape:
    return None
  sizeExpr = attr.shape.size()
  expr = str(sizeExpr.value).replace('@self','obj')
  if expr.find('@')>=0:
    # we cannot use this expression, it probably has @config in it.
    # we will use the shape_method, hopefully part of the Python interface
    if not attr.shape_method:
      return None
    expr = 'obj.%s()[0]' % attr.shape_method
  return expr

def categorize(ptype):
  '''Categorize the attributes and methods for dumping to a string.
  Returns an object called  psana_type with the following lists:

    psana_type.one_line_methods
    psana_type.multi_line_methods
    psana_type.list_multi_line_methods
    psana_type.idx_list_one_line_methods
    psana_type.idx_list_multi_line_methods

  Each member of each list is a method, or attribute, that should be printed.

  member format is as follows:

  one_line_methods = [ ('name', 'expression', 'type_to_str_function'), ... ]
     examples would be:
     ('num', 'num', 'int8_to_str')    for a pubilc attribute 'num' with no accessor
     ('num', 'num()', 'int8_to_str')  for a method 'num', with the expression to invoke it
     ('name', 'name()', 'str_to_str') for a null terminated string
  
  multi_line_methods = [ ('name', 'expression', 'type_to_str_function'), ...]
    for things that should get a newline after name and before printing

  list_one_line_methods = [ ('name', 'elem_type_to_str_function'), ...
    # for a method that returns a list
    ('pvControls','Control_pvControl_to_str'), ...
    for idx, subobj in enumerate(obj.pvControls()):
       ln += 'pvControls[%d]\n'%idx
       ln += Control_pvControl_to_str(subobj)
  idx_list_one_line_methods = [ ('name, 'elem_type_to_str_function', 'maxexpression'), ...
     # this is for a short list of things that print small, like:
    ('quads, 'uint32_to_str', 'quads_shape()[0]') 
    ln = 'quads'
    for idx in range( obj.quads_shape()[0] ):
      subobj = obj.quads(idx)
      ln += ' [%d]=%s' % (idx,uint32_to_str(subobj))
 
  idx_list_multi_line_methods
    # this is for a list of things that are complex, so we want a newline before each element
    ('vert', 'DataDescrV1_to_str', 'vert_shape()[0]') 
    for idx in range( obj.quads_shape()[0] ):
      ln = 'vert[%d]\n' % idx
      subobj = obj.vert(idx)
      ln += ' %s' % DataDescrV1_to_str(subobj)
    '''
  ####################
  # helper functions
  def methodInfo(method, arrstr, namespace, ptype):
    shapeStr = ''
    if method.attribute:
      attr = method.attribute
      shapeStr = 'shape=%s' % str(attr.shape)
    return "%s%s <- %s.%s.%s( %s ) value_type=%r %s"  % (method.type.name, arrstr,
                                                         namespace, ptype.name, method.name, method.args,
                                                         method.type.value_type, shapeStr)

  def classifyMethod(method):
    # args
    args = False
    if method.args is not None:
      if len(method.args)>0:
        args = True

    # basic (not basic includes char)
    basic = isBasicType(method.type)

    # array
    if (method.rank is None) or (method.rank == 0):
      array = False
      arrstr = ''
    else:
      array = True
      arrstr = '[%s]' % method.rank

    return basic, args, array, arrstr

  def handle_basic_args_not_array(namespace, ptype, method, arrstr, psana_type):
    rule = getBasicNotArrayRule(namespace, ptype.name, method.name)
    if rule is None:
      print "notProcessed 110: %s" % methodInfo(method, arrstr, namespace, ptype)
      return
    cmd, sub = rule
    if cmd == 'skip':
      return
    if cmd == '0_to_method':
      psana_type.idx_list_one_line_methods.append((method.name, 
                                                   getDumpFunction(method.type),
                                                   sub))
      return
    if cmd == '0_to_const':
      psana_type.idx_list_one_line_methods.append((method.name, 
                                                   getDumpFunction(method.type),
                                                   sub))
      return
    print "notProcessed 110: %s" % methodInfo(method, arrstr, namespace, ptype)

  def handle_not_basic_not_args_array(namespace, ptype, method, arrstr, 
                                      psana_type, methodReturnTypeNamespace):
    if method.rank != 1:
      print "notProcessed 001: %s" % methodInfo(method, arrstr, namespace, ptype)
      return
    if method.type.name == 'char':
      psana_type.one_line_methods.append((method.name, 'obj.%s()' % method.name, 'str_to_str'))
      return
    if isinstance(method.type, psddl.Enum.Enum):
      sizeExpr = getSizeExpr(method)
      if not sizeExpr:
        if method.name == 'capacitorValues' and namespace == 'Ipimb' and ptype.name.startswith('ConfigV'):
          sizeExpr = 4
        else:
          print "notProcessed 001: enum but no sizeExpr  %s" % (methodInfo(method, arrstr, namespace, ptype),)
          return
      psana_type.idx_list_one_line_methods.append((method.name, 
                                                   getDumpFunction(method.type),
                                                   sizeExpr))
      return
    # if the returned type is a value_type, the Python interface will have the method return a 
    # a list of the object. 
    # If it is not a value type, then the method will get an index argument.
    if not method.attribute:
      print "notProcessed 001: method has no attribute: %s" % methodInfo(method, arrstr, namespace, ptype)
      return
    attr = method.attribute
    if attr.type.value_type:
      psana_type.list_multi_line_methods.append((method.name, 
                                                 getDumpFunction(method.type, method.type.package.name)))
    else:
      # the method will take an index argument. Need to determine expression for last element.
      sizeExpr = getSizeExpr(method)
      if not sizeExpr:
        print "notProcessed 001: no sizeExpr %s" % methodInfo(method, arrstr, namespace, ptype)
        return
      psana_type.idx_list_multi_line_methods.append((method.name, 
                                                     getDumpFunction(method.type, methodReturnTypeNamespace),
                                                     sizeExpr))
 
  ###################
  # code
  psana_type = PsanaType()

  accessors = [attrib.accessor  \
               for attrib in ptype.attributes() \
               if attrib.accessor and (attrib.accessor.access=='public') \
               and (not attrib.accessor.name.startswith('_'))]

  accessorNames = [accessor.name for accessor in accessors]

  methodsThatAreNotAccessors = [method \
                                for method in ptype.methods() \
                                if (not method.name.startswith('_')) and \
                                (method.access=='public') and (method.name not in accessorNames)]
      
  methods = accessors + methodsThatAreNotAccessors

  basic_attr = getBasicAttrNoAccessor(ptype)
  psana_type.one_line_methods = [ (name,name,fn) for name,fn in basic_attr ]
  psana_type.multi_line_methods = []
  psana_type.list_multi_line_methods = []
  psana_type.idx_list_one_line_methods = []
  psana_type.idx_list_multi_line_methods = []

  namespace = ptype.package.name

  for method in methods:
    basic, args, array, arrstr = classifyMethod(method)
    # These flags, basic, args, array (all True/False) give 8 categories of 
    # methods. We deal with all 8 cases, a comment like 000 indicates which one

    methodReturnTypeNamespace = getMethodReturnTypeNamespace(method)
    # categorize
    if basic == False and args == False and array == False:   #  000
      psana_type.multi_line_methods.append((method.name,
                                            'obj.%s()' % method.name, 
                                            getDumpFunction(method.type, 
                                                            methodReturnTypeNamespace)))
                                           
    elif basic == True and args == False and array == False:  #  100
      psana_type.one_line_methods.append((method.name,
                                          'obj.%s()' % method.name, 
                                          getDumpFunction(method.type)))

    elif basic == False and args == True and  array == False: #  010
      print "notProcessed 010: %s" % methodInfo(method, arrstr, namespace, ptype)

    elif basic == True and args == True and array == False:   #  110
      handle_basic_args_not_array(namespace, ptype, method, arrstr, psana_type)

    elif basic == False and args == True and array == True:   #  011
      print "notProcessed 011: %s" % methodInfo(method, arrstr, namespace, ptype)

    elif basic == True and args == True and array == True:    #  111
      print "notProcessed 111: %s" % methodInfo(method, arrstr, namespace, ptype)

    elif basic == False and args == False and array  == True: #  001
      handle_not_basic_not_args_array(namespace, ptype, method, arrstr, 
                                      psana_type, methodReturnTypeNamespace)

    elif basic and args == False and array == True:           #  101
      fnCvt = getNdarrayDumpFunction(method.type)
      assert fnCvt, "did not get valid convert function for %r"  %method
      psana_type.one_line_methods.append((method.name, 
                                          'obj.%s()' % method.name,
                                          fnCvt))
   
  for x in psana_type.one_line_methods:
    assert len(x)==3, "one_line_method is not len 3: %r" % x
  for x in psana_type.multi_line_methods:
    assert len(x)==3, "multi_line_method is not len 3: %r" % x
  for x in psana_type.idx_list_one_line_methods:
    assert len(x)==3, "idx_list_one_line_method is not len 3: %r" % x
  for x in psana_type.idx_list_multi_line_methods:
    assert len(x)==3, "one_line_multi_line_method is not len 3: %r" % x
  for x in psana_type.list_multi_line_methods:
    assert len(x)==2, "list_multi_line_method is not len 2: %r" % x

  return psana_type
  
#  In case someone decides to run this module
#
usage = '''Module is not supposed to be run as a main module.
Invoke using psddlc from the psddl package. Run from a 
release directory where both the psddldata and the psana-tools
packages are checked out. Run as:

psddlc -I data -B package_dir:psana_test -b psana_test data/psddldata/*.ddl
'''

if __name__ == "__main__" :
  sys.exit ( usage )
