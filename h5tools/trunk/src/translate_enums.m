function dataSet = translate_enums(dataSet,fileName,dataSetName)
  % translates enums to their symbolic names in a dataset
  % ARGS IN:
  %  dataSet     - a datset, as read by h5read.
  %  fileName    - the filename this dataset was read from
  %  dataSetName - the name of this dataset within the h5 file
  % OUT:
  %  if the dataSet is a compound type, looks at the type of each field.
  %  If the type is enum, it replaces that column in the dataset with a 
  %  1D cell array of strings.  If the type is an array, checks if the 
  %  base type of the array is an enum.  If so replaces the array with a 
  %  cell array of strings.
  %
  %  Note: if an enum holds a value that cannot be translated, an error will
  %        occur.  
  %
  % WARNING: don't call this function twice on the same dataset. It assumes
  %          all fields in the struct are arrays.  After calling it once
  %          some fields may be cell arrays of strings.
  % $Id$

  fid = H5F.open(fileName);
  dsetid = H5D.open(fid, dataSetName);
  dtypeid = H5D.get_type(dsetid);
  if (H5T.get_class(dtypeid) ~= H5ML.get_constant_value('H5T_COMPOUND'))
    H5T.close(dtypeid);
    H5D.close(dsetid);
    H5F.close(fid);
    return
  end
  numCompoundFields = H5T.get_nmembers(dtypeid);  
  for j = 0:numCompoundFields-1
     fieldClass = H5T.get_member_class(dtypeid,j);     
     if (fieldClass == H5ML.get_constant_value('H5T_ENUM'))
       enumTypeId = H5T.get_member_type(dtypeid,j);
       enumInt2Name = get_enumInt2Name(enumTypeId);
       fieldName = H5T.get_member_name(dtypeid,j);
       fldNamesAsE = arrayfun(@(x) sprintf('e%d',x), getfield(dataSet, fieldName),'UniformOutput',false);
       newFieldVals = cellfun(@(x) getfield(enumInt2Name, x), fldNamesAsE, 'UniformOutput',false);
       dataSet = setfield(dataSet, fieldName, newFieldVals);
       H5T.close(enumTypeId);
     elseif (fieldClass == H5ML.get_constant_value('H5T_ARRAY'))
       arrayTypeId = H5T.get_member_type(dtypeid,j);
       baseTypeId = H5T.get_super(arrayTypeId);
       if (H5T.get_class(baseTypeId) == H5ML.get_constant_value('H5T_ENUM'))
         enumInt2Name = get_enumInt2Name(baseTypeId);
         fieldName = H5T.get_member_name(dtypeid,j);
         fldNamesAsE = arrayfun(@(x) sprintf('e%d',x), getfield(dataSet, fieldName),'UniformOutput',false);
         newFieldVals = cellfun(@(x) getfield(enumInt2Name, x), fldNamesAsE, 'UniformOutput',false);
         dataSet = setfield(dataSet, fieldName, newFieldVals);
       H5T.close(baseTypeId);
       H5T.close(arrayTypeId);
      end
     end
  end
  H5T.close(dtypeid);
  H5D.close(dsetid);
  H5F.close(fid);
end

function enumInt2Name = get_enumInt2Name(enumTypeId)
  % returns a struct mapping enum int's to symbolic names.  ints have e prepended to be variable names  
  % IN:  enumTypeId -  as returned by H5T
  % OUT: if the enum is ONE=1, FIVE=5
  %      the struct will look like
  %      struct.e1='ONE', struct.e5='FIVE'

  enumInt2Name = struct();
  enumNumberElements = H5T.get_nmembers(enumTypeId);
  for k = 0:enumNumberElements-1
    curEnumName = H5T.get_member_name(enumTypeId,k);
    curEnumInt = H5T.get_member_value(enumTypeId,k);
    curEnumIntFldName = sprintf('e%d',curEnumInt);
    enumInt2Name = setfield(enumInt2Name, curEnumIntFldName, curEnumName);
  end
end

