#ifndef TRANSLATOR_DSETTYPEPOS_H
#define TRANSLATOR_DSETTYPEPOS_H

#include "hdf5/hdf5.h"

namespace Translator {

/**
 * @ingroup Translator
 * 
 * @brief adds typeId to DataSetPos
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @author David Schneider
 */
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
