#ifndef TRANSLATOR_HDFWRITERDAMAGE_H
#define TRANSLATOR_HDFWRITERDAMAGE_H

#include <map>

#include "hdf5/hdf5.h"

#include "hdf5pp/Group.h"
#include "pdsdata/xtc/Damage.hh"

#include "Translator/DataSetPos.h"
#include "Translator/DataSetCreationProperties.h"
#include "Translator/HdfWriterGeneric.h"

namespace Translator {

/**
 *  @ingroup Translator
 *
 *  @brief class to write the damage datasets into an hdf5 group.
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @author David Schneider
 */
class HdfWriterDamage {
 public:
  typedef enum {BlankEntry=0, ValidEntry=1} MaskVal;
  HdfWriterDamage();
  ~HdfWriterDamage();

  void make_datasets(hid_t groupId);
  void make_datasets(hdf5pp::Group & group) { make_datasets(group.id()); }

  void append(hid_t groupId, Pds::Damage damage, MaskVal  mask) { 
    store_at(-1,groupId,damage,mask); 
  }
  void append(hdf5pp::Group & group, Pds::Damage damage, MaskVal mask) { 
    append(group.id(), damage, mask); 
  }

  void store_at(long index, hid_t groupId, Pds::Damage damage, MaskVal mask);
  void store_at(long index, hdf5pp::Group & group, Pds::Damage damage, MaskVal mask) { 
    store_at(index,group.id(),damage, mask); 
  }

  void closeDatasets(hid_t groupId);
  void closeDatasets(hdf5pp::Group &group) { closeDatasets(group.id()); }

  const DataSetCreationProperties & dataSetCreationProperties() 
  { return m_dataSetCreationProperties; }
  void setDatasetCreationProperties(const DataSetCreationProperties & dataSetCreationProperties) 
  { m_dataSetCreationProperties = dataSetCreationProperties; }


 private:
  DataSetCreationProperties m_dataSetCreationProperties;
  HdfWriterGeneric m_writer;
  size_t m_damagePos, m_maskPos;
  hid_t m_h5DamageTypeId;
};

} // namespace

#endif 
