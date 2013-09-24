//--------------------------------------------------------------------------
// File and Version Information:
//      $Id$
//
// Description:
//      Hand-written supporting types for DDL-HDF5 mapping.
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psddl_hdf2psana/xtc.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <cstring>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/EnumType.h"
#include "hdf5pp/Utils.h"
#include "PSEvt/DataProxy.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

// special class to facilitate construction of Pds::Src
struct SrcFactory: Pds::Src {
  SrcFactory(uint32_t log, uint32_t phy) {
    this->_log = log;
    this->_phy = phy;
  }
};

}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

namespace Pds {

hdf5pp::Type ns_ClockTime_v0_dataset_data_stored_type()
{
  typedef ns_ClockTime_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("seconds", offsetof(DsType, seconds), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("nanoseconds", offsetof(DsType, nanoseconds), hdf5pp::TypeTraits<uint32_t>::stored_type());
  return type;
}

hdf5pp::Type ns_ClockTime_v0::dataset_data::stored_type()
{
  static hdf5pp::Type type = ns_ClockTime_v0_dataset_data_stored_type();
  return type;
}

hdf5pp::Type ns_ClockTime_v0_dataset_data_native_type()
{
  typedef ns_ClockTime_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("seconds", offsetof(DsType, seconds), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("nanoseconds", offsetof(DsType, nanoseconds), hdf5pp::TypeTraits<uint32_t>::native_type());
  return type;
}

hdf5pp::Type ns_ClockTime_v0::dataset_data::native_type()
{
  static hdf5pp::Type type = ns_ClockTime_v0_dataset_data_native_type();
  return type;
}







hdf5pp::Type ns_DetInfo_v0_dataset_data_stored_type()
{
  typedef ns_DetInfo_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("processId", offsetof(DsType, processId), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("detector", offsetof(DsType, detector), hdf5pp::TypeTraits<const char*>::stored_type());
  type.insert("device", offsetof(DsType, device), hdf5pp::TypeTraits<const char*>::stored_type());
  type.insert("detId", offsetof(DsType, detId), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("devId", offsetof(DsType, devId), hdf5pp::TypeTraits<uint32_t>::stored_type());
  return type;
}

hdf5pp::Type ns_DetInfo_v0::dataset_data::stored_type()
{
  static hdf5pp::Type type = ns_DetInfo_v0_dataset_data_stored_type();
  return type;
}

hdf5pp::Type ns_DetInfo_v0::dataset_data::native_type()
{
  static hdf5pp::Type type = ns_DetInfo_v0_dataset_data_stored_type();
  return type;
}

ns_DetInfo_v0::dataset_data::dataset_data()
  : detector(0)
  , device(0)
{
}

ns_DetInfo_v0::dataset_data::dataset_data(const ::Pds::DetInfo& di)
  : processId(di.processId())
  , detector(0)
  , detId(di.detId())
  , device(0)
  , devId(di.devId())
{
  detector = strdup(::Pds::DetInfo::name(di.detector()));
  device = strdup(::Pds::DetInfo::name(di.device()));
}

ns_DetInfo_v0::dataset_data::dataset_data(const dataset_data& di)
  : processId(di.processId)
  , detector(0)
  , detId(di.detId)
  , device(0)
  , devId(di.devId)
{
  detector = strdup(di.detector);
  device = strdup(di.device);
}

ns_DetInfo_v0::dataset_data::~dataset_data()
{
  free(detector);
  free(device);
}

ns_DetInfo_v0::dataset_data&
ns_DetInfo_v0::dataset_data::operator=(const dataset_data& di)
{
  if (this != &di) {
    free(detector);
    free(device);
    processId = di.processId;
    detId = di.detId;
    detector = strdup(di.detector);
    devId = di.devId;
    device = strdup(di.device);
  }
  return *this;
}

ns_DetInfo_v0::dataset_data::operator ::Pds::DetInfo() const
{
  // get id from a string
  Pds::DetInfo::Detector det = Pds::DetInfo::NoDetector;
  for (int i = 0; i != int(Pds::DetInfo::NumDetector); ++ i) {
    const char* name = Pds::DetInfo::name(Pds::DetInfo::Detector(i));
    if (std::strcmp(name, detector) == 0) {
      det = Pds::DetInfo::Detector(i);
      break;
    }
  }
  Pds::DetInfo::Device dev = Pds::DetInfo::NoDevice;
  for (int i = 0; i != int(Pds::DetInfo::NumDevice); ++ i) {
    const char* name = Pds::DetInfo::name(Pds::DetInfo::Device(i));
    if (std::strcmp(name, device) == 0) {
      dev = Pds::DetInfo::Device(i);
      break;
    }
  }

  return ::Pds::DetInfo(processId, det, detId, dev, devId);
}




hdf5pp::Type ns_Src_v0_dataset_data_stored_type()
{
  typedef ns_Src_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("log", offsetof(DsType, log), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("phy", offsetof(DsType, phy), hdf5pp::TypeTraits<uint32_t>::stored_type());
  return type;
}

hdf5pp::Type ns_Src_v0::dataset_data::stored_type()
{
  static hdf5pp::Type type = ns_Src_v0_dataset_data_stored_type();
  return type;
}

hdf5pp::Type ns_Src_v0_dataset_data_native_type()
{
  typedef ns_Src_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("log", offsetof(DsType, log), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("phy", offsetof(DsType, phy), hdf5pp::TypeTraits<uint32_t>::native_type());
  return type;
}

hdf5pp::Type ns_Src_v0::dataset_data::native_type()
{
  static hdf5pp::Type type = ns_Src_v0_dataset_data_native_type();
  return type;
}

ns_Src_v0::dataset_data::operator ::Pds::Src() const
{
  return ::SrcFactory(log, phy);
}


} // namespace Pds
