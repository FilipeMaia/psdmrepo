#ifndef TRANSLATOR_DATASETMETA_H
#define TRANSLATOR_DATASETMETA_H

#include <string>
#include "Translator/DataSetPos.h"

namespace Translator {

class DataSetMeta : public DataSetPos {
  /**
   * @brief adds dataset typeid and name to dataset tracking information
   *
   * extends DataSetPos to also keep track of the hdf5 typeid and the name of the
   * dataset.  Assumes no ownership of the typeid, will not close it.
   */
public:
  DataSetMeta() {};
 DataSetMeta(const std::string & name, hid_t dsetId, MaxSize maxSize, hid_t typeId) : 
  DataSetPos(dsetId, maxSize), m_name(name), m_typeId(typeId) {}
  hid_t typeId() const { return m_typeId; }
  std::string name() const { return m_name; }
  
 private:
  std::string m_name;
  hid_t m_typeId;
};

} // namespace
#endif
