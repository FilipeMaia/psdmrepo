#--------------------------------------------------------------------------
# File and Version Information:
# $Id$
#
# Description:
#  Provides print function to interactively explore LCLS datasets
#  in h5py

#------------------------------------------------------------------------

'''
This module provides the function printds to print a h5py dataset.
This function is designed to work with a datasets of a compound type 
- a type like a C struct with several fields.
  
@version $Id$

@author David Schneider
'''

import h5py

######### User Interface Function ######################################

def printds(ds, rows=None, fields=None, skipfields=None, formatdict=None):
  '''
  prints a h5py dataset that is a scalar compound type, or a 1D array of
  a compound type.
  
  ARGS:
    ds   - the h5py dataset.
    rows - The default, None, means read all the rows of ds. Set rows
           to an increasing list (zero-up) of rows to print a subset of
           the dataset.
    fields - a list of integers that index fields, or a list of strings that
             name strings.  
    skipfields - a list of integers or a list of strings that name fields to skip
    formatdict - the default formatting is to apply %s to each cell.  To use
             different formating for certain fields, pass a dictionary here. 
             For example format={'serialID':'0x%8.8X'}, if the dataset has a field 
             called serialID, it will now print in hex.

  The output has three header lines:
  line 1:  Identifies any vlen data in fields
  line 2:  The field names for the data
  line 3:  The types for each field

  The first column is the row of the dataset.

  For example, if the dataset ds had the fields 'a' and 'b', one could do

  >>>printds(ds)  # see the whole dataset
  >>>printds(ds,fields=['a'])              # just see field 'a' (one column)
  >>>printds(ds,fields=[0])                # same thing as above
  >>>printds(ds,fields=[0,1])              # see both columns
  >>>printds(ds,skipfields=['a'])          # just see column 'b'
  >>>printds(ds,skipfields=[0])            # same as above
  >>>printds(ds,rows=[1,3],fields=['a'])   # only see two rows of column a
  >>>printds(ds,formatdict={'a':'%x'})     # column 'a' will print as hex
  '''
  ####### helper functions - private to printds
  def checkDs(ds):
    if not looksLikeH5pyDataset(ds):
      raise ValueError("Argument ds must be a h5py high level Dataset")

    isCompound = ds.dtype.names is not None
    if not isCompound:
      raise ValueError("Argument ds must be a dataset of a compound type")


  def getFieldNamesToPrint(ds,fields,skipfields):
    allFieldNames = [name for name in ds.dtype.names]
    if fields and skipfields:
      raise ValueError("Only one of fields or skipfields may be specified")

    fieldNames = allFieldNames
    if fields:
      if isListOfInts(fields):
        rangeCheck(fields,len(allFieldNames))
        fieldNames = [allFieldNames[i] for i in fields]
      elif isListOfStrs(fields):
        checkThatFieldsExist(fields,allFieldNames)
        fieldNames = fields
      else:
        raise ValueError("fields is neither list of ints or str")

    elif skipfields:
      if isListOfInts(skipfields):
        rangeCheck(skipfields,len(allFieldNames))
        fieldIndexes = [ i for i in range(len(allFieldNames)) if i not in skipfields]
        fieldNames = [allFieldNames[i] for i in fieldIndexes]
      elif isListOfStrs(skipfields):
        checkThatFieldsExist(skipfields,allFieldNames)
        fieldNames = [fld for fld in allFieldNames if fld not in skipfields]
      else:
        raise ValueError("fields is neither list of ints or str")
    return fieldNames, allFieldNames

  def setUpFormatDict(formatdict,allFieldNames):
    if formatdict:
      checkThatFieldsExist(formatdict.keys(), allFieldNames)
      for fld in fieldNames:
        if fld not in formatdict.keys():
          formatdict[fld]="%s"
    else:
      formatdict = dict(zip(fieldNames,["%s" for fld in fieldNames]))
    return formatdict

  def initializeFromRows(ds,rows,datasetIsScalar,datasetIsArray):
    columns = []
    multiline = []
    if datasetIsScalar:
      if rows:
        print "warning: dataset is scalar, ignoring rows"
        rows = None
      dsArray = ds
      rowIdxCells = ['','','','']
    elif datasetIsArray:
      if rows:
        dsArray = ds[rows]
      else:
        dsArray = ds[...]
        rows = range(len(ds))
      rowIdxCells = ['','rowIdx',''] + map(str,rows)
    else:
      raise Exception("ds is neither scalar nor 1D array, array dim is %d" % len(ds.shape))
    columns.append(rowIdxCells)
    multiline.append([1 for r in rowIdxCells])
    return columns,multiline,dsArray,rows

  def identifyField(ds,fld):
    fldDtype = ds.dtype[fld]
    enumDict = h5py.check_dtype(enum=fldDtype)
    vlenType = h5py.check_dtype(vlen=fldDtype)
    if len(fldDtype.shape)>1:
      raise ValueError("field %s is an array of dimension greater than 1.  Data is to complex." % fld)
    fldIsArray = len(fldDtype.shape)==1
    if fldIsArray:
      baseEnumDict = h5py.check_dtype(enum=fldDtype.base)
      baseDtype = fldDtype.base.name
    else:
      baseEnumDict = None
      baseDtype = None
    return fldDtype, enumDict, vlenType, fldIsArray, baseEnumDict, baseDtype

  def getSimplePrintData(datasetIsScalar, ds, fld, formatdict):
    if datasetIsScalar:
      printData =  [ formatdict[fld] % dsArray[fld] ]
    else:
      fldData = dsArray[fld]
      printData = map( lambda(x): formatdict[fld] % x, fldData )
    return printData

  ############## begin function
  checkDs(ds)
  fieldNames, allFieldNames = getFieldNamesToPrint(ds,fields,skipfields)
  if len(fieldNames)==0:
    return
  formatdict = setUpFormatDict(formatdict,allFieldNames)

  datasetIsArray = len(ds.shape)==1
  datasetIsScalar = len(ds.shape)==0

  columns,multiline,dsArray,rows = initializeFromRows(ds,rows,datasetIsScalar,datasetIsArray)

  for fld in fieldNames:
    fldDtype, enumDict, vlenType, fldIsArray,  baseEnumDict, baseDtype = identifyField(ds,fld)
    if enumDict is None and (vlenType is None or vlenType is str) and baseEnumDict is None and not fldIsArray:
      if vlenType is None:
        typeStr = str(fldDtype)
      else:
        typeStr = 'str'
      headerRows = ['', fld, typeStr]
      printData = headerRows + getSimplePrintData(datasetIsScalar, ds, fld, formatdict)
      columns.append(printData)
      multiline.append([1 for x in printData])

    elif enumDict and vlenType is None and baseEnumDict is None and not fldIsArray:
      fieldValues,types = getEnumValues(dsArray,datasetIsArray,fld,enumDict)
      if set([int,str]) == types:
        print "warning: dset=%s enum field=%s some values not in the enumeration" % \
          (ds.name, fieldName)
      printData = ['',fld,'enum'] + fieldValues
      columns.append(printData)
      multiline.append([1 for x in printData])
      
    elif fldIsArray and (vlenType is None or vlenType is str):
      arrayColumns = []
      havePrintedEnumTypeWarning = False
      for arrayColumn in range(fldDtype.shape[0]):
        if not baseEnumDict:
          baseNameForPrint = baseDtype
          if datasetIsArray:
            columnData = dsArray[fld][:,arrayColumn]
          else:
            columnData = [ dsArray[fld][arrayColumn] ]
          arrayColumns.append(map(lambda(x): formatdict[fld] % x, columnData))
        elif baseEnumDict:
          baseNameForPrint = 'enum'
          columnData,types = getEnumValues(dsArray,datasetIsArray,fld,baseEnumDict,arrayColumn)
          if set([int,str]) == types and not havePrintedEnumTypeWarning:
            print "warning: dset=%s array of enum field=%s some values not in the enumeration" % \
              (ds.name, fieldName)
            havePrintedEnumTypeWarning = True
          arrayColumns.append(columnData)
      joinedArrayColumn = joinColumns(', ',arrayColumns)
      joinedArrayColumn = [ '[' + x + ']' for x in joinedArrayColumn]
      printData = ['',fld,'array of %s' % baseNameForPrint] + joinedArrayColumn
      columns.append(printData)
      multiline.append([1 for x in printData])

    elif vlenType is not None and vlenType is not str and enumDict is None and baseEnumDict is None and not fldIsArray:
      vlenRecordFieldNames = vlenType.names
      vlenRecordFieldTypes = [vlenType[subFld].name for subFld in vlenRecordFieldNames]
      linesPerRec = [1,1]  # subflds, subtypes
      vlenColumns = [ [subFld,str(subType)] for subFld,subType in \
                      zip(vlenRecordFieldNames,vlenRecordFieldTypes)]
      if datasetIsScalar:
        vlenData = [ ds[fld] ]
      else:
        # better to access vlen data through dataset rather then ds[...] or dsArray as above
        vlenData = [ds[r,fld] for r in rows]
      for rec in vlenData:
        linesPerRec.append(len(rec))
        for vlenColIdx,subFld in enumerate(vlenRecordFieldNames):
          vlenColumns[vlenColIdx].extend(map(lambda(x): formatdict[fld] % x, rec[subFld]))
      joinedVlenColumn = joinColumns(', ',vlenColumns)
      joinedVlenColumn = joinedVlenColumn[0:2] + [ '(' + x + ')' for x in joinedVlenColumn[2:]]
      joinedWithBrackets = ['  ' + x for x in joinedVlenColumn[0:2]]
      current = 2
      for lineCount in linesPerRec[2:]:
        recLines = joinedVlenColumn[current:current+lineCount]
        current += lineCount
        if lineCount == 0:
          joinedWithBrackets.append('[]')
        elif lineCount == 1:
          joinedWithBrackets.append('[' + recLines[0] + ']')
        else:
          joinedWithBrackets.append('[' + recLines[0] + ',')
          for j in range(1,lineCount-1):
            joinedWithBrackets.append(' ' + recLines[j] + ',')
          joinedWithBrackets.append(' ' + recLines[-1] + ']')
      topLine = '%s (vlen)' % fld
      topLine = topLine.rjust(len(joinedWithBrackets[-1]))
      joinedWithBrackets.insert(0,topLine)
      linesPerRec.insert(0,1)
      columns.append(joinedWithBrackets)
      multiline.append(linesPerRec)

    else:
      raise ValueError("Data is to complex for this print function")

  printColumns = joinColumns(' ',columns, multiline)
  for row in printColumns:
    print row

##############################
# Helper functions

def looksLikeH5pyDataset(ds):
  return hasattr(ds,'name') and hasattr(ds,'dtype') and \
    hasattr(ds.dtype,'names') and hasattr(ds,'shape')


def isListOfInts(x):
  def isInstanceInt(x):
    return isinstance(x,int)
  return all(map(isInstanceInt,x))

def isListOfStrs(x):
  def isInstanceStr(x):
    return isinstance(x,str)
  return all(map(isInstanceStr,x))

def rangeCheck(intList,n):
  minVal,maxVal = min(intList),max(intList)
  if minVal < 0:
    raise IndexError("integer list include negative values")
  if maxVal > n-1:
    raise IndexError("integer list include a value, %d, greater than last index which is %d"  % (maxVal,n-1))

def checkThatFieldsExist(fields,allFieldNames):
  for fld in fields:
    if fld not in allFieldNames:
      raise IndexError("fields include '%s' which is not in dataset fieldnames: %r" % (fld,allFieldNames))

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
    enumValues = [ invertedEnumDict.get(x,int(x)) ]
    types = set([type(enumValues)])
  return enumValues,types

def joinColumns(joinStr, columns, multiline=None):
  '''When multiline is None - takes a list of columns and produces one column, merging
  entries line by line and right justifying each subcolumn to overall width of that subcolumn.
  Example:
  >>>joinColumns(("-",[ ['aa','bb'],['z','xx'] ])
  ['aa- z', 'bb-xx']
  Note the extra space in front of the z in the output, since the second column, ['z','xx'] has
  a maximum width of 2, everything in that column is right justified to 2 before joining.
  One would print each enry of the joined columns to get nicely formatted data:
    aa- z
    bb-xx
  When multiline is not none, each column may have several entries per row of the
  merged column.
  Example:
  >>>joinColumns("-", [ ['a1','a2', 'b1','b2','b3'], ['z1', 'xx1','xx2'] ], [ [2,3], [1,2] ])
  ['a1- z1', 'a2-   ', 'b1-xx1', 'b2-xx2', 'b3-   ']
  This will print at
    a1- z1
    a2-   
    b1-xx1
    b2-xx2
    b3-   
  The multline parameter specifies that we take the first 2 entries from column 0 and the first 
  entry from column 1 to make the 2 lines that form 'row' 1 of the merged column.  Likewise we
  take the next three from column 0 and the next 2 from column 1 to produce the next three lines
  for 'row' 2 of the merged column.
  '''
  widths = map(max,[map(len,col) for col in columns])
  if multiline is None:
    joinedColumn = []
    for row in range(len(columns[0])):
      rowCells = [column[row] for column in columns]
      widthCells = [cell.rjust(width) for width,cell in zip(widths,rowCells)]
      joinedColumn.append(joinStr.join(widthCells))
    return joinedColumn
  else:
    multilineLens = map(len,multiline)
    assert all([multilineLens[0] == x for x in multilineLens]), "multiline entries not the same across columns"
    assert len(multiline) == len(columns), "len of multiline list must equal len of column list"
    paddedColumns = [ [] for col in columns ]
    current = [0 for col in columns]
    for multiLineRow in range(multilineLens[0]):
      linesThisMultiRow = max([multiline[i][multiLineRow] for i in range(len(columns))])
      for colIdx,col,width in zip(range(len(columns)),columns,widths):
        linesThisCol = multiline[colIdx][multiLineRow]
        blankLines = linesThisMultiRow - linesThisCol
        for i in range(linesThisCol):
          cellData = col[current[colIdx]+i]
          paddedColumns[colIdx].append(cellData.rjust(width))
        for i in range(blankLines):
          paddedColumns[colIdx].append(' '.rjust(width))
        current[colIdx] += linesThisCol
    return joinColumns(joinStr,paddedColumns,None)
