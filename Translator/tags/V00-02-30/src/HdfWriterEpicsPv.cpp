#include <string>

#include "boost/make_shared.hpp"

#include "MsgLogger/MsgLogger.h"

#include "Translator/HdfWriterEpicsPv.h"
#include "Translator/epics.ddl.h"

using namespace Translator;
using namespace std;

namespace {

const string logger(const string addTo="") { 
  static const string base("Translator.HdfWriterEpicsPv");
  if (addTo.size()>0) {
    return base + std::string(".") + addTo;
  }
  return base;
}

} // namespace

std::ostream & Translator::operator<<(std::ostream & o, HdfWriterEpicsPv::DispatchAction da) {
  if (da==HdfWriterEpicsPv::CreateWriteClose) o << "CreateWriteClose";
  else if (da==HdfWriterEpicsPv::CreateAppend) o << "CreateAppend";
  else if (da ==HdfWriterEpicsPv::Append) o << "Append";
  else o << "**unknown**";
  return o;
}

HdfWriterEpicsPv::HdfWriterEpicsPv(const DataSetCreationProperties & oneElemDataSetCreationProperties,
                                   const DataSetCreationProperties & manyElemDataSetCreationProperties,
                                   boost::shared_ptr<HdfWriterEventId> hdfWriterEventId) 
  : m_oneElemDataSetCreationProperties(oneElemDataSetCreationProperties),
    m_manyElemDataSetCreationProperties(manyElemDataSetCreationProperties),
    m_hdfWriterEventId(hdfWriterEventId)
{
  m_hdfWriterGeneric = boost::make_shared<HdfWriterGeneric>("epics");
}

void HdfWriterEpicsPv::closeDataset(hid_t groupId) { 
  m_hdfWriterGeneric->closeDatasets(groupId); 
  m_hdfWriterEventId->closeDataset(groupId); 
}

HdfWriterEpicsPv::~HdfWriterEpicsPv() {
}



