#ifndef PSANA_TEST_EPICS2STR_H
#define PSANA_TEST_EPICS2STR_H

#include <string>
#include "pdsdata/xtc/Xtc.hh"
#include "pdsdata/psddl/epics.ddl.h"

namespace psana_test {

/*-----------
  If the xtc typeid is Id_Epics and the payload is undamaged, returns
  a string printing out the epics. Otherwise, returns an empty string.
 */
std::string epics2Str(Pds::Xtc *xtc);

std::string epicsPvHeader2Str(const Pds::Epics::EpicsPvHeader &pv);
std::string epicsPvCtrlString2str(const Pds::Epics::EpicsPvCtrlString &pv);
std::string epicsPvCtrlShort2str(const Pds::Epics::EpicsPvCtrlShort &pv);
std::string epicsPvCtrlFloat2str(const Pds::Epics::EpicsPvCtrlFloat &pv);
std::string epicsPvCtrlEnum2str(const Pds::Epics::EpicsPvCtrlEnum &pv);
std::string epicsPvCtrlChar2str(const Pds::Epics::EpicsPvCtrlChar &pv);
std::string epicsPvCtrlLong2str(const Pds::Epics::EpicsPvCtrlLong &pv);
std::string epicsPvCtrlDouble2str(const Pds::Epics::EpicsPvCtrlDouble &pv);
std::string epicsPvTimeString2str(const Pds::Epics::EpicsPvTimeString &pv);
std::string epicsPvTimeShort2str(const Pds::Epics::EpicsPvTimeShort &pv);
std::string epicsPvTimeFloat2str(const Pds::Epics::EpicsPvTimeFloat &pv);
std::string epicsPvTimeEnum2str(const Pds::Epics::EpicsPvTimeEnum &pv);
std::string epicsPvTimeChar2str(const Pds::Epics::EpicsPvTimeChar &pv);
std::string epicsPvTimeLong2str(const Pds::Epics::EpicsPvTimeLong &pv);
std::string epicsPvTimeDouble2str(const Pds::Epics::EpicsPvTimeDouble &pv);


} // namespace psana_test
#endif
