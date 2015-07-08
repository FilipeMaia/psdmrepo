#include "MsgLogger/MsgLogger.h"
#include "hdf5/hdf5.h"
#include "hdf5pp/Exceptions.h"
#include "Translator/HdfWriterDamage.h"

using namespace Translator;

namespace {

  struct  DamageStruct { 
    DamageStruct(Pds::Damage damage) 
      : bits(damage.bits())
      , DroppedContribution((damage.bits() & (1 << Pds::Damage::DroppedContribution)) != 0)
      , OutOfOrder((damage.bits() & (1 << Pds::Damage::OutOfOrder)) != 0)
      , OutOfSynch((damage.bits() & (1 << Pds::Damage::OutOfSynch)) != 0)
      , UserDefined((damage.bits() & (1 << Pds::Damage::UserDefined)) != 0)
      , IncompleteContribution((damage.bits() & (1 << Pds::Damage::IncompleteContribution)) != 0)
      , userBits(damage.userBits())
    {}
    uint32_t bits;
    uint8_t DroppedContribution;
    uint8_t OutOfOrder;
    uint8_t OutOfSynch;
    uint8_t UserDefined;
    uint8_t IncompleteContribution;
    uint8_t userBits;
  };

  const char * logger = "HdfWriterDamage";
};

HdfWriterDamage::HdfWriterDamage() : m_writer("damage")
{
  m_damagePos = 0xFFFF;
  m_maskPos = 0xFFFF;
  m_h5DamageTypeId = H5Tcreate(H5T_COMPOUND, sizeof(DamageStruct));
  herr_t status = std::min(0, H5Tinsert(m_h5DamageTypeId, "bits", offsetof(DamageStruct,bits), H5T_NATIVE_UINT32));
  status = std::min(status, H5Tinsert(m_h5DamageTypeId, "DroppedContribution", offsetof(DamageStruct,DroppedContribution), H5T_NATIVE_UINT8));
  status = std::min(status, H5Tinsert(m_h5DamageTypeId, "OutOfOrder", offsetof(DamageStruct,OutOfOrder), H5T_NATIVE_UINT8));
  status = std::min(status, H5Tinsert(m_h5DamageTypeId, "OutOfSynch", offsetof(DamageStruct,OutOfSynch), H5T_NATIVE_UINT8));
  status = std::min(status, H5Tinsert(m_h5DamageTypeId, "UserDefined", offsetof(DamageStruct,UserDefined), H5T_NATIVE_UINT8));
  status = std::min(status, H5Tinsert(m_h5DamageTypeId, "IncompleteContribution", offsetof(DamageStruct,IncompleteContribution), H5T_NATIVE_UINT8));
  status = std::min(status, H5Tinsert(m_h5DamageTypeId, "userBits", offsetof(DamageStruct,userBits), H5T_NATIVE_UINT8));
  if ((m_h5DamageTypeId < 0) or (status !=0)) MsgLog(logger,fatal,"unable to create Damage compound type");
  MsgLog(logger,trace,"created hdf5 type for _damage= " << m_h5DamageTypeId);
}

void HdfWriterDamage::make_datasets(hid_t groupId)
{
  try {
    m_damagePos = m_writer.createUnlimitedSizeDataset(groupId,
                                                      "_damage",
                                                      m_h5DamageTypeId,m_h5DamageTypeId,
                                                      dataSetCreationProperties());
    m_maskPos = m_writer.createUnlimitedSizeDataset(groupId,
                                                    "_mask",
                                                    H5T_NATIVE_UINT8, H5T_NATIVE_UINT8,
                                                    dataSetCreationProperties());
  } catch (ErrSvc::Issue &issue) {
    std::ostringstream msg;
    msg << "HdfWriterDamage - make_dataset failed for _damage or _mask. "
        << " Generic writer failure: " << issue.what();
    throw HdfWriterGeneric::DataSetException(ERR_LOC, msg.str());
  }
  if ((m_damagePos != 0) or (m_maskPos != 1)) {
    throw HdfWriterGeneric::DataSetException(ERR_LOC,
             "HdfWritereDamage::make_datasets, damagePos != 0 or maskPos != 1");
  }
}

void HdfWriterDamage::store_at(long index, hid_t groupId, Pds::Damage damage, MaskVal mask) 
{
  if ((m_damagePos != 0) or (m_maskPos != 1)) {
    throw HdfWriterGeneric::DataSetException(ERR_LOC,
               "HdfWritereDamage::store_at, damagePos != 0 or maskPos != 1");
  }

  DamageStruct damageBuffer(damage);
  uint8_t maskBuffer = mask;

  try {
    m_writer.store_at(groupId, index, m_damagePos, &damageBuffer);
    m_writer.store_at(groupId, index, m_maskPos, &maskBuffer);
  } catch (ErrSvc::Issue &issue) {
    std::ostringstream msg;
    msg << "HdfWriterDamage::store_at - writer failure: " << issue.what();
    throw HdfWriterGeneric::WriteException(ERR_LOC,msg.str());
  }
}

void HdfWriterDamage::closeDatasets(hid_t groupId)
{
  try {
    m_writer.closeDatasets(groupId);
  } catch (ErrSvc::Issue &issue) {
    std::ostringstream msg;
    msg << "HdfWriterDamage - failed to close " 
        << "dataset, writer failure: " << issue.what();
    throw HdfWriterGeneric::DataSetException(ERR_LOC, msg.str());
  }
}

HdfWriterDamage::~HdfWriterDamage()
{
  MsgLog(logger,debug,"HdfWriterDamage::~HdfWriterDamage()");
  herr_t status = H5Tclose(m_h5DamageTypeId);
  if (status<0) {
    std::ostringstream msg;
    msg << "Error with H5Tclose call for damage  type id = " << m_h5DamageTypeId;
    throw HdfWriterGeneric::DataSetException(ERR_LOC, msg.str());
  }
  MsgLog(logger,trace,"closed hdf5 type for _damage (" << m_h5DamageTypeId << ")");

}
