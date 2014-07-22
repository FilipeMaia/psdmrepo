#include "XtcInput/DgramList.h"

namespace XtcInput {

void DgramList::push_back(const XtcInput::Dgram & dg) {
  m_dgramList.push_back(dg.dg());
  m_filenameList.push_back(dg.file());
}

Dgram::ptr DgramList::frontDg() const {
  return m_dgramList.at(0);
}

} // namespace XtcInput
