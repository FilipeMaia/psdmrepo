#ifndef TRANSLATOR_DSETTYPEPOS_H
#define TRANSLATOR_DSETTYPEPOS_H

#include "hdf5/hdf5.h"

namespace Translator {

class DataSetTypePos : public DataSetPos {
public:
  DataSetTypePos() {};
  DataSetTypePos(hid_t dsetId, hid_t typeId) : DataSetPos(dsetId), m_typeId(typeId) {}
  hid_t typeId() { return m_typeId; }
 private:
  hid_t m_typeId;
};

} // namespace
#endif
