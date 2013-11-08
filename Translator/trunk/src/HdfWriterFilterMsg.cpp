#include <string.h>

#include "MsgLogger/MsgLogger.h"
#include "PSTime/Time.h"

#include "hdf5/hdf5.h"
#include "hdf5pp/Exceptions.h"

#include "Translator/HdfWriterFilterMsg.h"

using namespace Translator;

namespace {

  const char * logger = "HdfWriterFilterMsg";
};

HdfWriterFilterMsg::HdfWriterFilterMsg() {
  m_dsetPos = 0xFFFF;
  herr_t err = m_h5typeId = H5Tcopy(H5T_C_S1);
  err = std::min(err,H5Tset_size(m_h5typeId, H5T_VARIABLE));
  if (err<0) throw HdfWriterGeneric::DataSetException(ERR_LOC,
        "HdfWriterFilterMsg - error making variable string type, H5Tset_size of H5TCopy problem");
  MsgLog(logger,trace,"Created type for filter msg: " << m_h5typeId);
}

HdfWriterFilterMsg::~HdfWriterFilterMsg() {
  herr_t status = H5Tclose(m_h5typeId);
  if (status<0) throw HdfWriterGeneric::DataSetException(ERR_LOC,
                "error closing type in ~HdfWriterFilterMsg");
  MsgLog(logger,trace,"closed type for filter msg (" << m_h5typeId << ")");
}

void HdfWriterFilterMsg::make_dataset(hid_t groupId)
{
  try {
    m_dsetPos = m_writer.createUnlimitedSizeDataset(groupId,
                                                    "message",
                                                    m_h5typeId,
                                                    dataSetCreationProperties());
  } catch (ErrSvc::Issue &issue) {
    std::ostringstream msg;
    msg << "HdfWriterFilterMsg - make_dataset failed. Generic writer failure: " << issue.what();
    throw HdfWriterGeneric::DataSetException(ERR_LOC, msg.str());
  }
  if (m_dsetPos != 0) throw HdfWriterGeneric::DataSetException(ERR_LOC, 
                            "HdfWriterFilterMsg::make_dataset - dsetPos is not 0");

}

void HdfWriterFilterMsg::append(hid_t groupId, const std::string & msg) 
{  
  if (m_dsetPos != 0) throw HdfWriterGeneric::WriteException(ERR_LOC, 
                            "HdfWriterFilterMsg::append - no dataset created, dsetPos is not 0");

  typedef const char *ConstCharPtr;
  ConstCharPtr wdata[1];
  wdata[0] = msg.c_str();

  try {
    m_writer.append(groupId, m_dsetPos, wdata);
  } catch (ErrSvc::Issue &issue) {
    std::ostringstream msg;
    msg << "HdfWriterEventId::append failed to write filter msg = " 
        << msg
        << " - generic writer failed: " << issue.what();
    throw HdfWriterGeneric::WriteException(ERR_LOC,msg.str());
  }
}


void HdfWriterFilterMsg::closeDataset(hid_t groupId)
{
  try {
    m_writer.closeDatasets(groupId);
  } catch (ErrSvc::Issue &issue) {
    std::ostringstream msg;
    msg << "HdfWriterFilterMsg - failed to close " 
        << "dataset, writer failure: " << issue.what();
    throw HdfWriterGeneric::DataSetException(ERR_LOC, msg.str());
  }
}
