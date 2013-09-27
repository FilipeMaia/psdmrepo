#--------------------------------------------------------------------------
# File and Version Information:
# $Id$
#
# Description:
#  Wrapper code for h5py to make datasets with complex dtypes easier to work

#------------------------------------------------------------------------

'''High level wrapper code to work with h5py datasets with many fields

This module provides two functions to work with datasets returned by
h5py.  These functions work with datasets of a compound type - a type like
a C struct with several fields.  The two functions are:
  
tablePrint - prints a tabular view of the dataset

flds2attrs - returns an object where each field of the compound type is a
             separate attribute.

Note: both these functions read the entire dataset into memory.  Use the
optional rows parameter to read a subset of the data.

@version $Id$

@author David Schneider
'''

import h5py
import numpy as np


######### User Interface Functions ######################################

def flds2attrs(ds,rows=None,formatStrings=None,columnNames=None):
  '''
  Takes a h5py dataset of a compound type.  Returns an object where each field of 
  the compound is a separate attribute, or data member of the object.

  Optional arguments:
    rows - an increasing list of rows to pull.  Use to analyze a subset of
           the dataset or to deal with running out of memory for large datasets.
    formatStrings - when printing the objects in tabular form, you may want to
           use hex or scientific notation for different fields.  Index this by
           the field names, ie:
           formatStrings={'serialID':'0x%8.8X'}  if the dataset has a field 
           called serialID, it will now print in hex.
    columnNames - when printing in objects in tabular form, you can get very
           long lines for the bigger tables.  Use this to abbreviate column
           names.  For example: columnsNames={'serialID': 'sid'}

  For example, if ds is a h5py dataset that holds the following numpy array:

  array([(0, 1L, 2), (10, 12L, 13)], 
        dtype=[('f1', '<u2'), ('f2', '<u4'), ('f3', '|u1')])
 
   then flds2attrs(ds) will return an object we'll call obj with the following 
   attributes and values:

   obj.f1   a numpy arry of uint16  with values [0,10]
   obj.f2   a numpy arry of uint32  with values [1,12]
   obj.f3   a numpy arry of uint8   with values [2,13]

   and if one does

   print obj

   one will get:

         f1       f2      f3
   (uint16) (uint32) (uint8)
          0        1       2
         10       12      13

  If ds is a scalar dataset, obj.f1, obj.f2, and obj.f3 will be simple types, 
  not numpy arrays.

  The two features of this object are the tabular printing as above, and accessing
  each of the compound fields through a named attribute.

  Fields that are an enums will be translated into a Python list of Python strings (or
  left as an int if not in the enumeration).

  Fields that are an array of enums are separated by column.  For example, if the
  dataset contains the field 'capacitorValue' that is an array of 4 enums, the
  object will have the 4 attributes, 'capacitorValue0', 'capacitorValue1', etc.

  When using the optional argument columnNames for a field that is an array of enums, 
  just reference the orginal field name.  For example, adding 
  columnNames={'capacitorValue': 'capval'} 
  will cause 'capval0', 'capval1', ... to print as the headers for the attributes
  'capacitorValue0', 'capacitorValue1', ... .

  Fields that are vlen strings will be translated into a Python list of strings.

  Fields that are other vlen data are assumed to have a numpy array of a compound 
  type for the underlying data.  They are translated into a Python list of these 
  underlying numpy arrays.
  '''
  if not looksLikeH5pyDataset(ds):
    raise ValueError("Argument ds must be a h5py high level Dataset")

  isCompound = ds.dtype.names is not None
  if isCompound:
    return AttrsFromDs(ds,rows,formatStrings,columnNames)

  raise ValueError("Argument ds must be a dataset of a compound type")

def tablePrint(ds,rows=None,formatStrings=None,columnNames=None):
  '''Takes a h5py dataset of a compound type.  
  '''
  fld2attrObj = flds2attrs(ds,rows,formatStrings,columnNames)
  print fld2attrObj

######### Class Types Returned to User ################################
class AttrsFromDs(object):
  '''Wrapper class for a h5py dataset that is of a compound type
  Separates all fields of the compound into attributes.  
  Translate enums into strings.  If a field in the compound is an
  array of enum, it is split into multiple columns, one for each column of the
  enum.  A column that is vlen str will be mapped to a list of python strings.
  A column that is vlen is mapped to a list of the underlying vlen data (assumed to be 
  a numpy array with a compound dtype).

  IN:
    ds      - an h5py dataset for a compound type, as returned by h5py.
    rows    - if you don't want all rows for the dataset, pass a list (in increasing
              order) of the subset of rows that you want.  
              Defaults to None which means select all rows.
    formatStrings - optional - a dictionary format strings for certain fields.  I.e:
              {'calibration':'%.3e'} if you want the calibration data printed as such
              in the table view (what print displays).
    columnNames - optional - a dict of alternative names to use when printing the columns
               headers in the table view.  For example, to use abbreviations: 
               {'calibration':'cal'}  Default is the field names from the dataset.
  '''
  def __init__(self,ds,rows=None,formatStrings=None,columnNames=None):
    assert ds.dtype.names, "dataset is not compound"

    self._fieldNames = [name for name in ds.dtype.names]
    self._fieldTypes = dict([ (fieldName, ds.dtype[fieldName].name) for fieldName in ds.dtype.names ])
    self._vlenField = dict([(fieldName,False) for fieldName in ds.dtype.names ])
    datasetIsArray = len(ds.shape)==1
    datasetIsScalar = len(ds.shape)==0

    if datasetIsScalar:
      if rows:
        print "warning: dataset is scalar, ignoring rows"
        rows = None
      dsArray = ds
    elif datasetIsArray:
      if rows:
        dsArray = ds[rows]
      else:
        dsArray = ds[...]
    else:
      raise Exception("ds is neither scalar nor 1D array, array dim is %d" % len(ds.shape))

    self._datasetRows = rows
    self._datasetName = ds.name

    self._formatStrings = dict( [(name,'%s') for name in self._fieldNames])
    if formatStrings is not None:
      self._formatStrings.update(formatStrings)

    self._columnNames = dict([(name,name) for name in self._fieldNames])
    if columnNames is not None:
      self._columnNames.update(columnNames)

    self._enumsInDataset = {}

    fieldNamesToChange = []  # fields that are array's of enums will be split apart

    for fieldName in self._fieldNames:
      fldDtype = ds.dtype[fieldName]
      enumDict = h5py.check_dtype(enum=fldDtype)
      vlenType = h5py.check_dtype(vlen=fldDtype)
      fldIsArray = len(fldDtype.shape)==1
      if fldIsArray:
        baseEnumDict = h5py.check_dtype(enum=fldDtype.base)
      else:
        baseEnumDict = None

      if enumDict is None and vlenType is None and baseEnumDict is None:
        setattr(self, fieldName, dsArray[fieldName])

      elif enumDict:
        self._fieldTypes[fieldName]='enum'
        self._enumsInDataset[fieldName]=enumDict
        fieldValues,types = getEnumValues(dsArray,datasetIsArray,fieldName,enumDict)
        if set([int,str]) == types:
          print "warning: dset=%s enum field=%s some values not in the enumeration" % \
            (ds.name, fieldName)
        setattr(self,fieldName,fieldValues)

      elif baseEnumDict:
        self._enumsInDataset[fieldName]=baseEnumDict
        numberOfColumns = fldDtype.shape[0]
        fieldNamesToChange.append((fieldName,numberOfColumns))
        types = set([])
        for column in range(numberOfColumns):
          newFieldName = '%s%d'%(fieldName,column)
          colValues,colTypes = getEnumValues(dsArray,datasetIsArray,fieldName,
                                             baseEnumDict,column=column)
          types = types.union(colTypes)
          setattr(self,newFieldName,colValues)
        if set([int,str]) == types:
          print "warning: dset=%s 1D array of enum field=%s some values not in the enumeration" % \
            (ds.name, fieldName)

      elif vlenType is str:
        if datasetIsArray:
          setattr(self,fieldName,dsArray[fieldName].tolist())
        else:
          setattr(self,fieldName,str(dsArray[fieldName]))
        self._fieldTypes[fieldName]='str'

      elif vlenType:
        if not datasetIsArray:
          setattr(self,fieldName,ds[fieldName])
        else:
          # it is better to access vlen data as ds[0,fld] rather then dsArray[fld][0]
          if self._datasetRows:
            rowsToUse = self._datasetRows
          else:
            rowsToUse = range(dsArray.shape[0])
          setattr(self,fieldName,[ds[r,fieldName] for r in rowsToUse])
        vlenRecordFieldNames = vlenType.names
        vlenRecordFieldTypes = [vlenType[fld].name for fld in vlenRecordFieldNames]
        vlenRecordTypeString = ' '.join(['%s:%s' % (vlenFldName,vlenFldTypeName) for \
                                         vlenFldName,vlenFldTypeName in \
                                         zip(vlenRecordFieldNames, vlenRecordFieldTypes)])
        self._fieldTypes[fieldName]='vlen ' + vlenRecordTypeString
        self._vlenField[fieldName]=True
      else:
        raise Exception("Unexpected property in field %s of dataset %s" % (fieldName,ds.name))

    # replace array of enum fields with field-column names
    for fieldName,numberOfColumns in fieldNamesToChange:
      indexOfFieldName = self._fieldNames.index(fieldName)
      self._fieldNames.remove(fieldName)
      fmtString = self._formatStrings[fieldName]
      colName = self._columnNames[fieldName]
      del self._formatStrings[fieldName]
      del self._columnNames[fieldName]
      del self._fieldTypes[fieldName]
      del self._vlenField[fieldName]
      for column in range(numberOfColumns-1,-1,-1):
        fieldNameWithColumn = '%s%d'%(fieldName,column)
        self._fieldNames.insert(indexOfFieldName,fieldNameWithColumn)
        self._formatStrings[fieldNameWithColumn] = fmtString
        self._columnNames[fieldNameWithColumn] = '%s%d' % (colName,column)
        self._fieldTypes[fieldNameWithColumn] = 'enum'
        self._vlenField[fieldNameWithColumn] = False

  def __str__(self):
    formattingList = [{'attr':fld, \
                       'fmt':self._formatStrings[fld], \
                       'headerLines':[self._columnNames[fld], \
                                      '(%s)' % self._fieldTypes[fld]
                                     ], \
                       'footerLines':[], \
                       'vlen':self._vlenField[fld] } for \
                      fld in self._fieldNames]
    tableStr = attributesToTableStr(self,formatting=formattingList)
    if self._datasetRows:
      tableStr += 'rows: %s\n' % rangeString(self._datasetRows)
    if len(self._enumsInDataset)>0:
      tableStr += 'Enums:'
      for fld,enum in self._enumsInDataset.iteritems():
        tableStr += '\n%s: %r' % (fld,enum)
    tableStr = 'dataset: ' + self._datasetName + '\n' + tableStr + '\n'
    return tableStr

  def __repr__(self):
    return str(self)


###### Helper Functions ###############################################

def looksLikeH5pyDataset(ds):
  return hasattr(ds,'name') and hasattr(ds,'dtype') and \
    hasattr(ds.dtype,'names') and hasattr(ds,'shape')

def getEnumValues(dsSource,arrayField,fieldName,enumDict,column=None):
  '''extracts an enum field, or column from a field that is an enum array
  and uses the enumDict to translate it to strings.  values that are not in the 
  enum dict are returned as a Python int (as opposed to the native numpy type). 
  Returns a list or value depending on arrayField.  Returns a set of the types 
  after the conversion as well.  The set of types will be set([str]) if all enums
  were converted to strings, or set([str,int]) if some of the enums could not be
  translated.
  IN
    dsSource     needs to support dsSource[fieldName] to get at the enum values
    arrayField   indicates that dsSource[fieldName] will be an array, not a scalar
    fieldName    
    enumDict  
    column    set to an integer to indicate that fieldName is a 1D array of enum
              as opposed to an enum.  This will pull out dsSource[fieldName][column] or 
              dsSource[fieldName][:,column] if arrayField is True
  OUT
    enumValues  usually a list of strings (if arrayField=True) or a scalar value
         types  set of all types returns - str,int or str, or possibly just int
  '''
  invertedEnumDict = dict([(intVal, strVal) for strVal,intVal in enumDict.iteritems()])
  if arrayField:
    if column is None:
      enumValues = [invertedEnumDict.get(x,int(x)) for x in dsSource[fieldName] ]
    else:
      enumValues = [invertedEnumDict.get(x,int(x)) for x in dsSource[fieldName][:,column] ]
    types = set(map(type,enumValues))
  else:
    if column is None:
      x = dsSource[fieldName]
    else:
      x = dsSource[fieldName][column]
    enumValues = invertedEnumDict.get(x,int(x))
    types = set([type(enumValues)])
  return enumValues,types

def attributesToTableStr(obj,formatting):
  '''Creates a printable table of object attributes.
  IN
    obj        - the object with attributes to put in the table
    formatting - a dict with: {'attr': name of attr to print in a column
                               'fmt': something like '%s', could also be '0x%X' or '%.e', etc
                               'headerLines': something like ['columnname','(uint16)'], these appear
                                              at the top of the column for the attribute
                               'footerLines': a list of lines that appear at the bottom of the column
                               'vlen': True if this is vlen data}
  OUT
    str
  '''
  if len(formatting)==0:
    return ''
  columns = []
  for fieldInfo in formatting:
    fieldName = fieldInfo['attr']
    formatString =  fieldInfo['fmt']
    headerLines = fieldInfo['headerLines']
    footerLines = fieldInfo['footerLines']
    vlen = fieldInfo['vlen']
    # This function works generically on an object where the attributes are arrays, or scalars. 
    # All attributes of a scalar object are simple types, unless an attribute is vlen data.  To 
    # Identify vlen data coming from a scalar type, we require the vlen key in the formatting dict
    # above.

    fieldValues = getattr(obj,fieldName)

    if isinstance(fieldValues,str):
      fieldStrings = [fieldValues]
    elif vlen and isinstance(fieldValues,np.ndarray):
      fieldStrings = [str(fieldValues)]
    else:
      try:
        fieldStrings = map(lambda x: formatString % x, fieldValues)
      except TypeError:
        # fieldName is not an iteratable
        fieldStrings = [formatString % fieldValues]

    columns.append(headerLines + fieldStrings + footerLines)

  columnLens = map(len,columns)
  assert all([columnLens[0] == colLen for colLen in columnLens]), \
    "Not all object attributes have the same lengths.  Lens are %s" % columnLens

  columnWidths = map( max, [map(len,column) for column in columns])
  table = ''
  row = 0
  while row < columnLens[0]:
    cells = [column[row] for column in columns]
    table += ' '.join([cell.rjust(width) for width,cell in zip(columnWidths,cells)])
    table += '\n'
    row += 1

  return table

def rangeString(vals):
  '''returns a string to print the values in a list of integers.
  Prints runs of consecutive values as a range.  For example,
  rangeString([1,2,3,6,7,8]) returns '1-3,6-8'
  It is assumed assumes values are increasing
  '''
  if len(vals)==0:
    return ''  
  s = str(vals[0])
  for i in range(1,len(vals)):
    if vals[i] - vals[i-1] > 1:
      if s[-1] ==  '-':
        s += '%s,%s' % (vals[i-1],vals[i])
      else:
        s += ',%s' % vals[i]
    else:
      if s[-1] != '-':
        s +=  '-'
      if i == len(vals)-1:
        s += '%s' % vals[i]
  return s

