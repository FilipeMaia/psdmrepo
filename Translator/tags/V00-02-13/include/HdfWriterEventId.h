#ifndef TRANSLATOR_HDFWRITEREVENTID_H
#define TRANSLATOR_HDFWRITEREVENTID_H

#include <map>

#include "hdf5/hdf5.h"

#include "hdf5pp/Group.h"
#include "PSEvt/EventId.h"

#include "Translator/DataSetPos.h"
#include "Translator/DataSetCreationProperties.h"
#include "Translator/HdfWriterGeneric.h"

namespace Translator {

/**
 *  @ingroup Translator
 *
 *  @brief class to write the eventId or time dataset into  hdf5 groups.
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @author David Schneider
 */
class HdfWriterEventId {
 public:
  HdfWriterEventId();
  ~HdfWriterEventId();

  void make_dataset(hdf5pp::Group & group);
  void make_dataset(hid_t groupId);
  void append(hdf5pp::Group & group, const PSEvt::EventId & eventId);
  void append(hid_t groupId, const PSEvt::EventId & eventId);

  const DataSetCreationProperties & dataSetCreationProperties() 
  { return m_dataSetCreationProperties; }
  void setDatasetCreationProperties(const DataSetCreationProperties & dataSetCreationProperties) 
  { m_dataSetCreationProperties = dataSetCreationProperties; }

  void closeDataset(hdf5pp::Group &group);
  void closeDataset(hid_t groupId);
 private:
  DataSetCreationProperties m_dataSetCreationProperties;
  hid_t m_h5typeId;
  HdfWriterGeneric m_writer;
  size_t m_dsetPos;
};

} // namespace

#endif 
