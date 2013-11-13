#ifndef PSDDL_HDF2PSANA_XTC_H
#define PSDDL_HDF2PSANA_XTC_H 1

//--------------------------------------------------------------------------
// File and Version Information:
//      $Id$
//
// Description:
//      Hand-written supporting types for DDL-HDF5 mapping.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Group.h"
#include "hdf5pp/Type.h"
#include "pdsdata/xtc/DetInfo.hh"
#include "pdsdata/xtc/ClockTime.hh"

namespace Pds {

namespace ns_ClockTime_v0 {
struct dataset_data {

  dataset_data() {}
  dataset_data(const ::Pds::ClockTime& clock)
    : seconds(clock.seconds()), nanoseconds(clock.nanoseconds()) {}

  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  uint32_t seconds;
  uint32_t nanoseconds; 

  operator ::Pds::ClockTime() const { return ::Pds::ClockTime(seconds, nanoseconds); }
};
} // namespace ns_ClockTime_v0



namespace ns_DetInfo_v0 {
struct dataset_data {

  dataset_data();
  dataset_data(const ::Pds::DetInfo& di);
  dataset_data(const dataset_data&);
  ~dataset_data();

  dataset_data& operator=(const dataset_data&);

  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  uint32_t processId;
  char* detector;
  uint32_t detId;
  char* device;
  uint32_t devId;

  operator ::Pds::DetInfo() const ;
};
} // namespace ns_DetInfo_v0


namespace ns_Src_v0 {
struct dataset_data {

  dataset_data() {}
  dataset_data(const ::Pds::Src& src) : log(src.log()), phy(src.phy()) {}

  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  uint32_t log;
  uint32_t phy;

  operator ::Pds::Src() const ;
};
} // namespace ns_DetInfo_v0



} // namespace Pds

#endif // PSDDL_HDF2PSANA_XTC_H
