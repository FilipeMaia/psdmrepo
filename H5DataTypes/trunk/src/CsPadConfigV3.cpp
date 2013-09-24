//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadConfigV3...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/CsPadConfigV3.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/ArrayType.h"
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

CsPadProtectionSystemThreshold::CsPadProtectionSystemThreshold(const Pds::CsPad::ProtectionSystemThreshold& o)
  : adcThreshold(o.adcThreshold())
  , pixelCountThreshold(o.pixelCountThreshold())
{
}

hdf5pp::Type
CsPadProtectionSystemThreshold::native_type()
{
  hdf5pp::CompoundType protSysType = hdf5pp::CompoundType::compoundType<CsPadProtectionSystemThreshold>() ;
  protSysType.insert_native<uint32_t>( "adcThreshold", offsetof(CsPadProtectionSystemThreshold, adcThreshold) ) ;
  protSysType.insert_native<uint32_t>( "pixelCountThreshold", offsetof(CsPadProtectionSystemThreshold, pixelCountThreshold) ) ;
  return protSysType;
}


CsPadConfigV3::CsPadConfigV3 ( const XtcType& data )
  : concentratorVersion(data.concentratorVersion())
  , runDelay(data.runDelay())
  , eventCode(data.eventCode())
  , protectionEnable(data.protectionEnable())
  , inactiveRunMode(data.inactiveRunMode())
  , activeRunMode(data.activeRunMode())
  , testDataIndex(data.tdi())
  , payloadPerQuad(data.payloadSize())
  , badAsicMask0(data.badAsicMask0())
  , badAsicMask1(data.badAsicMask1())
  , asicMask(data.asicMask())
  , quadMask(data.quadMask())
{
  for ( int q = 0; q < MaxQuadsPerSensor ; ++ q ) {
    roiMask[q] = data.roiMask(q);
  }
  
  for ( int q = 0; q < MaxQuadsPerSensor ; ++ q ) {
    quads[q] = data.quads(q);
  }

  const ndarray<const Pds::CsPad::ProtectionSystemThreshold, 1>& thresh = data.protectionThresholds();
  std::copy(thresh.begin(), thresh.end(), protectionThresholds);

  // fill the list of active sections from ROI bits
  int ns = 0 ;
  for ( int q = 0; q < MaxQuadsPerSensor ; ++ q ) {
    for ( int s = 0; s < SectionsPerQuad ; ++ s ) {
      sections[q][s] = -1;
      if ( roiMask[q] & (1<<s) ) sections[q][s] = ns++;
    }
  }
  
}

hdf5pp::Type
CsPadConfigV3::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
CsPadConfigV3::native_type()
{
  hsize_t sdims[2] = {MaxQuadsPerSensor, SectionsPerQuad};
  hdf5pp::Type baseSectType = hdf5pp::TypeTraits<int8_t>::native_type();
  hdf5pp::ArrayType sectArrType = hdf5pp::ArrayType::arrayType(baseSectType, 2, sdims);

  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<CsPadConfigV3>() ;
  confType.insert_native<uint32_t>( "concentratorVersion", offsetof(CsPadConfigV3, concentratorVersion) ) ;
  confType.insert_native<uint32_t>( "runDelay", offsetof(CsPadConfigV3, runDelay) ) ;
  confType.insert_native<uint32_t>( "eventCode", offsetof(CsPadConfigV3, eventCode) ) ;
  confType.insert("protectionThresholds", offsetof(CsPadConfigV3, protectionThresholds), CsPadProtectionSystemThreshold::native_type(), MaxQuadsPerSensor ) ;
  confType.insert_native<uint32_t>( "protectionEnable", offsetof(CsPadConfigV3, protectionEnable) ) ;
  confType.insert_native<uint32_t>( "inactiveRunMode", offsetof(CsPadConfigV3, inactiveRunMode) ) ;
  confType.insert_native<uint32_t>( "activeRunMode", offsetof(CsPadConfigV3, activeRunMode) ) ;
  confType.insert_native<uint32_t>( "testDataIndex", offsetof(CsPadConfigV3, testDataIndex) ) ;
  confType.insert_native<uint32_t>( "payloadPerQuad", offsetof(CsPadConfigV3, payloadPerQuad) ) ;
  confType.insert_native<uint32_t>( "badAsicMask0", offsetof(CsPadConfigV3, badAsicMask0) ) ;
  confType.insert_native<uint32_t>( "badAsicMask1", offsetof(CsPadConfigV3, badAsicMask1) ) ;
  confType.insert_native<uint32_t>( "asicMask", offsetof(CsPadConfigV3, asicMask) ) ;
  confType.insert_native<uint32_t>( "quadMask", offsetof(CsPadConfigV3, quadMask) ) ;
  confType.insert_native<uint32_t>( "roiMask", offsetof(CsPadConfigV3, roiMask) ) ;
  confType.insert("quads", offsetof(CsPadConfigV3, quads), CsPadConfigV1QuadReg::native_type(), MaxQuadsPerSensor ) ;
  confType.insert("sections", offsetof(CsPadConfigV3, sections), sectArrType ) ;

  return confType ;
}

void
CsPadConfigV3::store( const XtcType& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  CsPadConfigV3 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
