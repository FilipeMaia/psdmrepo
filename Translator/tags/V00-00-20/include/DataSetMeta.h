#ifndef TRANSLATOR_DATASETMETA_H
#define TRANSLATOR_DATASETMETA_H

#include <string>
#include "Translator/DataSetPos.h"

namespace Translator {

/**
 * @brief adds dataset typeid and name to dataset tracking information
 *
 * extends DataSetPos to also keep track of the hdf5 typeid and the name of the
 * dataset.  Assumes no ownership of the typeid, will not close it.
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @author David Schneider
 */
class DataSetMeta : public DataSetPos {
public:
  DataSetMeta() {};
 DataSetMeta(const std::string & name, hid_t dsetId, Shape shape, hid_t typeId) : 
  DataSetPos(dsetId, shape), m_name(name), m_typeId(typeId) {}
  hid_t typeId() const { return m_typeId; }
  std::string name() const { return m_name; }
  
 private:
  std::string m_name;
  hid_t m_typeId;
};

} // namespace
#endif
