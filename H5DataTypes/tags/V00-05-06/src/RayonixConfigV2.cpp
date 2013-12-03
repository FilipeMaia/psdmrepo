//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class RayonixConfigV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/RayonixConfigV2.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/ArrayType.h"
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/EnumType.h"
#include "hdf5pp/TypeTraits.h"
#include "H5DataTypes/H5DataUtils.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

RayonixConfigV2::RayonixConfigV2 ( const Pds::Rayonix::ConfigV2& data )
  : binning_f(data.binning_f())
  , binning_s(data.binning_s())
  , testPattern(data.testPattern())
  , exposure(data.exposure())
  , trigger(data.trigger())
  , rawMode(data.rawMode())
  , darkFlag(data.darkFlag())
  , readoutMode(data.readoutMode())
{
  strncpy(deviceID, data.deviceID(), 40);
}

hdf5pp::Type
RayonixConfigV2::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
RayonixConfigV2::native_type()
{
  hdf5pp::EnumType<uint32_t> _enum_type_readoutMode = hdf5pp::EnumType<uint32_t>::enumType();
  _enum_type_readoutMode.insert("Unknown", Pds::Rayonix::ConfigV2::Unknown);
  _enum_type_readoutMode.insert("Standard", Pds::Rayonix::ConfigV2::Standard);
  _enum_type_readoutMode.insert("HighGain", Pds::Rayonix::ConfigV2::HighGain);
  _enum_type_readoutMode.insert("LowNoise", Pds::Rayonix::ConfigV2::LowNoise);
  _enum_type_readoutMode.insert("HDR", Pds::Rayonix::ConfigV2::HDR);

  typedef RayonixConfigV2 DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("binning_f", offsetof(DsType, binning_f), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("binning_s", offsetof(DsType, binning_s), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("testPattern", offsetof(DsType, testPattern), hdf5pp::TypeTraits<int16_t>::native_type());
  type.insert("exposure", offsetof(DsType, exposure), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("trigger", offsetof(DsType, trigger), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("rawMode", offsetof(DsType, rawMode), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("darkFlag", offsetof(DsType, darkFlag), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("readoutMode", offsetof(DsType, readoutMode), _enum_type_readoutMode);
  type.insert("deviceID", offsetof(DsType, deviceID), hdf5pp::TypeTraits<const char*>::native_type(40));
  return type;
}

void
RayonixConfigV2::store( const Pds::Rayonix::ConfigV2& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  RayonixConfigV2 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
