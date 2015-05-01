#ifndef PSDDL_HDF2PSANA_SCHEMACONSTANTS_H
#define PSDDL_HDF2PSANA_SCHEMACONSTANTS_H

#include <string>

namespace psddl_hdf2psana {

  // name of the attribute holding schema version
  extern const std::string versionAttrName;

  // name of the attributes holding Src info
  extern const std::string srcAttrName;

  // name of the attributes holding Event Key 
  extern const std::string eventKeyAttrName;

  // name of the attributes holding H5Group name key 
  extern const std::string h5GroupNameKeyAttrName;

  // name of the attribute saying if this is a ndarray type
  extern const std::string ndarrayAttrName;

  // if ndarray, name of the attribute with dimension
  extern const std::string ndarrayDimAttrName;

  // if ndarray, name of the attribute with enum NDArrayParameters::ElemType 
  // describing type
  extern const std::string ndarrayElemTypeAttrName;

  // if ndarray, name of the attribute with elem Type size in bytes 
  extern const std::string ndarraySizeBytesAttrName;

  // if ndarray, name of the attribute saying if ndarray elem is const
  extern const std::string ndarrayConstElemAttrName;

  // if ndarray, name of the attribute saying if slow dim is vlen
  extern const std::string vlenAttrName;

  // h5 group names
  extern const std::string srcEventKeySeperator;

};

#endif
