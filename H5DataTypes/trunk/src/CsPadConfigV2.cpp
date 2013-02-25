//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadConfigV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/CsPadConfigV2.h"

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


CsPadConfigV2::CsPadConfigV2 ( const XtcType& data )
  : concentratorVersion(data.concentratorVersion())
  , runDelay(data.runDelay())
  , eventCode(data.eventCode())
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
    quads[q] = data.quads()[q];
  }

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
CsPadConfigV2::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
CsPadConfigV2::native_type()
{
  hsize_t sdims[2] = {CsPadConfigV2::MaxQuadsPerSensor, CsPadConfigV2::SectionsPerQuad};
  hdf5pp::Type baseSectType = hdf5pp::TypeTraits<int8_t>::native_type();
  hdf5pp::ArrayType sectArrType = hdf5pp::ArrayType::arrayType(baseSectType, 2, sdims);

  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<CsPadConfigV2>() ;
  confType.insert_native<uint32_t>( "concentratorVersion", offsetof(CsPadConfigV2, concentratorVersion) ) ;
  confType.insert_native<uint32_t>( "runDelay", offsetof(CsPadConfigV2, runDelay) ) ;
  confType.insert_native<uint32_t>( "eventCode", offsetof(CsPadConfigV2, eventCode) ) ;
  confType.insert_native<uint32_t>( "inactiveRunMode", offsetof(CsPadConfigV2, inactiveRunMode) ) ;
  confType.insert_native<uint32_t>( "activeRunMode", offsetof(CsPadConfigV2, activeRunMode) ) ;
  confType.insert_native<uint32_t>( "testDataIndex", offsetof(CsPadConfigV2, testDataIndex) ) ;
  confType.insert_native<uint32_t>( "payloadPerQuad", offsetof(CsPadConfigV2, payloadPerQuad) ) ;
  confType.insert_native<uint32_t>( "badAsicMask0", offsetof(CsPadConfigV2, badAsicMask0) ) ;
  confType.insert_native<uint32_t>( "badAsicMask1", offsetof(CsPadConfigV2, badAsicMask1) ) ;
  confType.insert_native<uint32_t>( "asicMask", offsetof(CsPadConfigV2, asicMask) ) ;
  confType.insert_native<uint32_t>( "quadMask", offsetof(CsPadConfigV2, quadMask) ) ;
  confType.insert_native<uint32_t>( "roiMask", offsetof(CsPadConfigV2, roiMask) ) ;
  confType.insert("quads", offsetof(CsPadConfigV2, quads), CsPadConfigV1QuadReg::native_type(), MaxQuadsPerSensor ) ;
  confType.insert("sections", offsetof(CsPadConfigV2, sections), sectArrType ) ;

  return confType ;
}

void
CsPadConfigV2::store( const XtcType& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  CsPadConfigV2 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
