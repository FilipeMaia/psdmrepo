#include "psddl_hdf2psana/SchemaConstants.h"

namespace psddl_hdf2psana {

  // attr names
  const std::string versionAttrName("_schemaVersion");
  const std::string srcAttrName("_xtcSrc");
  const std::string eventKeyAttrName("_eventKeyStr");
  const std::string h5GroupNameKeyAttrName("_groupKeyStr");
  const std::string ndarrayAttrName("_ndarray");
  const std::string ndarrayDimAttrName("_ndarrayDim");
  const std::string ndarrayElemTypeAttrName("_ndarrayElemType");
  const std::string ndarraySizeBytesAttrName("_ndarraySizeBytes");
  const std::string ndarrayConstElemAttrName("_ndarrayConstElem");
  const std::string vlenAttrName("_vlen");

  // h5 group names
  const std::string srcEventKeySeperator("__");
};
