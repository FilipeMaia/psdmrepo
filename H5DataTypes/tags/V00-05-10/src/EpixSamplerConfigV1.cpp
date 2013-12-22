//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpixSamplerConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/EpixSamplerConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/TypeTraits.h"
#include "H5DataTypes/H5DataUtils.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

EpixSamplerConfigV1::EpixSamplerConfigV1 ( const Pds::EpixSampler::ConfigV1& data )
  : version(data.version())
  , runTrigDelay(data.runTrigDelay())
  , daqTrigDelay(data.daqTrigDelay())
  , daqSetting(data.daqSetting())
  , adcClkHalfT(data.adcClkHalfT())
  , adcPipelineDelay(data.adcPipelineDelay())
  , digitalCardId0(data.digitalCardId0())
  , digitalCardId1(data.digitalCardId1())
  , analogCardId0(data.analogCardId0())
  , analogCardId1(data.analogCardId1())
  , numberOfChannels(data.numberOfChannels())
  , samplesPerChannel(data.samplesPerChannel())
  , baseClockFrequency(data.baseClockFrequency())
  , testPatternEnable(data.testPatternEnable())
{
}

hdf5pp::Type
EpixSamplerConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EpixSamplerConfigV1::native_type()
{
  typedef EpixSamplerConfigV1 DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("version", offsetof(DsType, version), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("runTrigDelay", offsetof(DsType, runTrigDelay), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("daqTrigDelay", offsetof(DsType, daqTrigDelay), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("daqSetting", offsetof(DsType, daqSetting), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("adcClkHalfT", offsetof(DsType, adcClkHalfT), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("adcPipelineDelay", offsetof(DsType, adcPipelineDelay), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("digitalCardId0", offsetof(DsType, digitalCardId0), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("digitalCardId1", offsetof(DsType, digitalCardId1), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("analogCardId0", offsetof(DsType, analogCardId0), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("analogCardId1", offsetof(DsType, analogCardId1), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("numberOfChannels", offsetof(DsType, numberOfChannels), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("samplesPerChannel", offsetof(DsType, samplesPerChannel), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("baseClockFrequency", offsetof(DsType, baseClockFrequency), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("testPatternEnable", offsetof(DsType, testPatternEnable), hdf5pp::TypeTraits<uint8_t>::native_type());
  return type;
}

void
EpixSamplerConfigV1::store( const Pds::EpixSampler::ConfigV1& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  EpixSamplerConfigV1 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
