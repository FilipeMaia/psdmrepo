#include <string.h>

#include "MsgLogger/MsgLogger.h"
#include "PSTime/Time.h"

#include "hdf5/hdf5.h"
#include "hdf5pp/Exceptions.h"

#include "Translator/HdfWriterString.h"

using namespace Translator;

namespace {

  const char * logger = "HdfWriterString";
};

HdfWriterString::HdfWriterString() : m_writer("string") {
  herr_t err = m_h5typeId = H5Tcopy(H5T_C_S1);
  err = std::min(err,H5Tset_size(m_h5typeId, H5T_VARIABLE));
  if (err<0) throw HdfWriterGeneric::DataSetException(ERR_LOC,
        "HdfWriterString - error making variable string type, H5Tset_size of H5TCopy problem");
  MsgLog(logger,trace,"Created hdf5 type for variable string: " << m_h5typeId);
}

HdfWriterString::~HdfWriterString() {
  herr_t status = H5Tclose(m_h5typeId);
  if (status<0) throw HdfWriterGeneric::DataSetException(ERR_LOC,
                "error closing type in ~HdfWriterString");
  MsgLog(logger,trace,"closed hdf5 type for variable string (" << m_h5typeId << ")");
}

void HdfWriterString::make_dataset(hid_t groupId)
{
  int dsetPos = 0xFF;
  try {
    dsetPos = m_writer.createUnlimitedSizeDataset(groupId,
                                                  datasetName,
                                                  m_h5typeId, m_h5typeId,
                                                  dataSetCreationProperties());
  } catch (ErrSvc::Issue &issue) {
    std::ostringstream msg;
    msg << "HdfWriterString - make_dataset failed. Generic writer failure: " << issue.what();
    throw HdfWriterGeneric::DataSetException(ERR_LOC, msg.str());
  }
  if (dsetPos != 0) throw HdfWriterGeneric::DataSetException(ERR_LOC, 
                                          "HdfWriterString::make_dataset - dsetPos is not 0");

}

void HdfWriterString::store(hid_t groupId, const std::string &msg)
{
  typedef const char *ConstCharPtr;
  ConstCharPtr wdata[1];
  wdata[0] = msg.c_str();
  
  try {
    m_writer.createAndStoreDataset(groupId,
                                   datasetName,
                                   m_h5typeId, 
                                   m_h5typeId,
                                   wdata);
  } catch (ErrSvc::Issue &issue) {
    std::ostringstream msg;
    msg << "HdfWriterString - storet failed. Generic writer failure: " << issue.what();
    throw HdfWriterGeneric::DataSetException(ERR_LOC, msg.str());
  }
}

void HdfWriterString::append(hid_t groupId, const std::string & msg) 
{  
  typedef const char *ConstCharPtr;
  ConstCharPtr wdata[1];
  wdata[0] = msg.c_str();

  try {
    m_writer.append(groupId, datasetName, wdata);
  } catch (ErrSvc::Issue &issue) {
    std::ostringstream msg;
    msg << "HdfWriterEventId::append failed to write filter msg = " 
        << msg
        << " - generic writer failed: " << issue.what();
    throw HdfWriterGeneric::WriteException(ERR_LOC,msg.str());
  }
}


void HdfWriterString::closeDataset(hid_t groupId)
{
  try {
    m_writer.closeDatasets(groupId);
  } catch (ErrSvc::Issue &issue) {
    std::ostringstream msg;
    msg << "HdfWriterString - failed to close " 
        << "dataset, writer failure: " << issue.what();
    throw HdfWriterGeneric::DataSetException(ERR_LOC, msg.str());
  }
}

const std::string HdfWriterString::datasetName("data");
