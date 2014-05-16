#include <sstream>

#include "MsgLogger/MsgLogger.h"
#include "PSTime/Time.h"

#include "hdf5/hdf5.h"
#include "hdf5pp/Exceptions.h"

#include "Translator/HdfWriterEventId.h"

using namespace Translator;

namespace {

  struct  EventIdStruct { 
    uint32_t seconds, nanoseconds, ticks, fiducials, control, vector;
  };

  const char * logger = "HdfWriterEventId";
};

HdfWriterEventId::HdfWriterEventId() : m_writer("eventId") {
  m_dsetPos = 0xFFFF;
  m_h5typeId = H5Tcreate(H5T_COMPOUND, sizeof(EventIdStruct));
  herr_t status = std::min(0,H5Tinsert(m_h5typeId, "seconds", offsetof(EventIdStruct,seconds), H5T_NATIVE_UINT32));
  status = std::min(status,H5Tinsert(m_h5typeId, "nanoseconds", offsetof(EventIdStruct,nanoseconds), H5T_NATIVE_UINT32));
  status = std::min(status,H5Tinsert(m_h5typeId, "ticks", offsetof(EventIdStruct,ticks), H5T_NATIVE_UINT32));
  status = std::min(status,H5Tinsert(m_h5typeId, "fiducials", offsetof(EventIdStruct,fiducials), H5T_NATIVE_UINT32));
  status = std::min(status,H5Tinsert(m_h5typeId, "control", offsetof(EventIdStruct,control), H5T_NATIVE_UINT32));
  status = std::min(status,H5Tinsert(m_h5typeId, "vector", offsetof(EventIdStruct,vector), H5T_NATIVE_UINT32));
  if ((m_h5typeId < 0) or (status < 0)) MsgLog(logger,fatal,"unable to create eventId compound type");
  MsgLog(logger,trace,"Created hdf5 type for EventId  " << m_h5typeId);
}

HdfWriterEventId::~HdfWriterEventId() {
  herr_t status = H5Tclose(m_h5typeId);
  if (status<0) MsgLog(logger,error,"error closing type");
  MsgLog(logger,trace,"Closed hdf5 type for EventId (" << m_h5typeId << ")");
}

void HdfWriterEventId::make_dataset(hdf5pp::Group & group)
{
  make_dataset(group.id());
}

void HdfWriterEventId::make_dataset(hid_t groupId)
{
  try {
    m_dsetPos = m_writer.createUnlimitedSizeDataset(groupId,
                                                    "time",
                                                    m_h5typeId, m_h5typeId,
                                                    dataSetCreationProperties());
  } catch (ErrSvc::Issue &issue) {
    std::ostringstream msg;
    msg << "HdfWriterEventId - make_dataset failed. Generic writer failure: " << issue.what();
    throw HdfWriterGeneric::DataSetException(ERR_LOC, msg.str());
  }
  if (m_dsetPos != 0) throw HdfWriterGeneric::DataSetException(ERR_LOC, 
                            "HdfWriterEventId::make_dataset - dsetPos is not 0");
}

void HdfWriterEventId::append(hdf5pp::Group & group, const PSEvt::EventId & eventId) 
{  
  append(group.id(), eventId);
}

void HdfWriterEventId::append(hid_t groupId, const PSEvt::EventId & eventId) 
{  
  if (m_dsetPos != 0) throw HdfWriterGeneric::WriteException(ERR_LOC, 
                            "HdfWriterEventId::append - no dataset created, dsetPos is not 0");

  EventIdStruct buffer;
  PSTime::Time time = eventId.time();
  buffer.seconds = time.sec();
  buffer.nanoseconds = time.nsec();
  buffer.ticks = eventId.ticks();
  buffer.fiducials = eventId.fiducials();
  buffer.control = eventId.control();
  buffer.vector = eventId.vector();

  try {
    m_writer.append(groupId, m_dsetPos, &buffer);
  } catch (ErrSvc::Issue &issue) {
    std::ostringstream msg;
    msg << "HdfWriterEventId::append failed to write eventId with sec = " 
        << time.sec() << " nano= " << time.nsec() 
        << " - generic writer failed: " << issue.what();
    throw HdfWriterGeneric::WriteException(ERR_LOC,msg.str());
  }
}

void HdfWriterEventId::closeDataset(hdf5pp::Group & group)
{
  closeDataset(group.id());
}

void HdfWriterEventId::closeDataset(hid_t groupId)
{
  try {
    m_writer.closeDatasets(groupId);
  } catch (ErrSvc::Issue &issue) {
    std::ostringstream msg;
    msg << "HdfWriterEventId - failed to close " 
        << "dataset, writer failure: " << issue.what();
    throw HdfWriterGeneric::DataSetException(ERR_LOC, msg.str());
  }
}
