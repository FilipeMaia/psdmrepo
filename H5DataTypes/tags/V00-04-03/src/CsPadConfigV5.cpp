//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadConfigV5...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/CsPadConfigV5.h"

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

CsPadConfigV3QuadReg::CsPadConfigV3QuadReg(const Pds::CsPad::ConfigV3QuadReg& src)
  : readClkSet(src.readClkSet())
  , readClkHold(src.readClkHold())
  , dataMode(src.dataMode())
  , prstSel(src.prstSel())
  , acqDelay(src.acqDelay())
  , intTime(src.intTime())
  , digDelay(src.digDelay())
  , ampIdle(src.ampIdle())
  , injTotal(src.injTotal())
  , rowColShiftPer(src.rowColShiftPer())
  , ampReset(src.ampReset())
  , digCount(src.digCount())
  , digPeriod(src.digPeriod())
  , biasTuning(src.biasTuning())
  , pdpmndnmBalance(src.pdpmndnmBalance())
  , readOnly(src.ro())
  , digitalPots(src.dp())
  , gainMap(*src.gm())
{
  const uint32_t* p = src.shiftSelect();
  std::copy(p, p+TwoByTwosPerQuad, this->shiftSelect);
  p = src.edgeSelect();
  std::copy(p, p+TwoByTwosPerQuad, this->edgeSelect);
}

hdf5pp::Type
CsPadConfigV3QuadReg::native_type()
{
  hdf5pp::CompoundType quadType = hdf5pp::CompoundType::compoundType<CsPadConfigV3QuadReg>() ;
  quadType.insert_native<uint32_t>( "shiftSelect", offsetof(CsPadConfigV3QuadReg, shiftSelect), TwoByTwosPerQuad) ;
  quadType.insert_native<uint32_t>( "edgeSelect", offsetof(CsPadConfigV3QuadReg, edgeSelect), TwoByTwosPerQuad) ;
  quadType.insert_native<uint32_t>( "readClkSet", offsetof(CsPadConfigV3QuadReg, readClkSet) ) ;
  quadType.insert_native<uint32_t>( "readClkHold", offsetof(CsPadConfigV3QuadReg, readClkHold) ) ;
  quadType.insert_native<uint32_t>( "dataMode", offsetof(CsPadConfigV3QuadReg, dataMode) ) ;
  quadType.insert_native<uint32_t>( "prstSel", offsetof(CsPadConfigV3QuadReg, prstSel) ) ;
  quadType.insert_native<uint32_t>( "acqDelay", offsetof(CsPadConfigV3QuadReg, acqDelay) ) ;
  quadType.insert_native<uint32_t>( "intTime", offsetof(CsPadConfigV3QuadReg, intTime) ) ;
  quadType.insert_native<uint32_t>( "digDelay", offsetof(CsPadConfigV3QuadReg, digDelay) ) ;
  quadType.insert_native<uint32_t>( "ampIdle", offsetof(CsPadConfigV3QuadReg, ampIdle) ) ;
  quadType.insert_native<uint32_t>( "injTotal", offsetof(CsPadConfigV3QuadReg, injTotal) ) ;
  quadType.insert_native<uint32_t>( "rowColShiftPer", offsetof(CsPadConfigV3QuadReg, rowColShiftPer) ) ;
  quadType.insert_native<uint32_t>( "ampReset", offsetof(CsPadConfigV3QuadReg, ampReset) ) ;
  quadType.insert_native<uint32_t>( "digCount", offsetof(CsPadConfigV3QuadReg, digCount) ) ;
  quadType.insert_native<uint32_t>( "digPeriod", offsetof(CsPadConfigV3QuadReg, digPeriod) ) ;
  quadType.insert_native<uint32_t>( "biasTuning", offsetof(CsPadConfigV3QuadReg, biasTuning) ) ;
  quadType.insert_native<uint32_t>( "pdpmndnmBalance", offsetof(CsPadConfigV3QuadReg, pdpmndnmBalance) ) ;
  quadType.insert("readOnly", offsetof(CsPadConfigV3QuadReg, readOnly), CsPadReadOnlyCfg::native_type()) ;
  quadType.insert("digitalPots", offsetof(CsPadConfigV3QuadReg, digitalPots), CsPadDigitalPotsCfg::native_type()) ;
  quadType.insert("gainMap", offsetof(CsPadConfigV3QuadReg, gainMap), CsPadGainMapCfg::native_type()) ;
  return quadType;
}


CsPadConfigV5::CsPadConfigV5 ( const XtcType& data )
  :  concentratorVersion(data.concentratorVersion())
  ,  runDelay(data.runDelay())
  ,  eventCode(data.eventCode())
  ,  protectionEnable(data.protectionEnable())
  ,  inactiveRunMode(data.inactiveRunMode())
  ,  activeRunMode(data.activeRunMode())
  ,  internalTriggerDelay(data.internalTriggerDelay())
  ,  testDataIndex(data.tdi())
  ,  payloadPerQuad(data.payloadSize())
  ,  badAsicMask0(data.badAsicMask0())
  ,  badAsicMask1(data.badAsicMask1())
  ,  asicMask(data.asicMask())
  ,  quadMask(data.quadMask())
{
  for ( int q = 0; q < MaxQuadsPerSensor ; ++ q ) {
    roiMask[q] = data.roiMask(q);
  }
  
  for ( int q = 0; q < MaxQuadsPerSensor ; ++ q ) {
    quads[q] = data.quads()[q];
  }

  for ( int q = 0; q < MaxQuadsPerSensor ; ++ q ) {
    protectionThresholds[q] = data.protectionThresholds()[q];
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
CsPadConfigV5::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
CsPadConfigV5::native_type()
{
  hsize_t sdims[2] = {MaxQuadsPerSensor, SectionsPerQuad};
  hdf5pp::Type baseSectType = hdf5pp::TypeTraits<int8_t>::native_type();
  hdf5pp::ArrayType sectArrType = hdf5pp::ArrayType::arrayType(baseSectType, 2, sdims);

  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<CsPadConfigV5>() ;
  confType.insert_native<uint32_t>( "concentratorVersion", offsetof(CsPadConfigV5, concentratorVersion) ) ;
  confType.insert_native<uint32_t>( "runDelay", offsetof(CsPadConfigV5, runDelay) ) ;
  confType.insert_native<uint32_t>( "eventCode", offsetof(CsPadConfigV5, eventCode) ) ;
  confType.insert("protectionThresholds", offsetof(CsPadConfigV5, protectionThresholds), CsPadProtectionSystemThreshold::native_type(), MaxQuadsPerSensor ) ;
  confType.insert_native<uint32_t>( "protectionEnable", offsetof(CsPadConfigV5, protectionEnable) ) ;
  confType.insert_native<uint32_t>( "inactiveRunMode", offsetof(CsPadConfigV5, inactiveRunMode) ) ;
  confType.insert_native<uint32_t>( "activeRunMode", offsetof(CsPadConfigV5, activeRunMode) ) ;
  confType.insert_native<uint32_t>( "internalTriggerDelay", offsetof(CsPadConfigV5, internalTriggerDelay) ) ;
  confType.insert_native<uint32_t>( "testDataIndex", offsetof(CsPadConfigV5, testDataIndex) ) ;
  confType.insert_native<uint32_t>( "payloadPerQuad", offsetof(CsPadConfigV5, payloadPerQuad) ) ;
  confType.insert_native<uint32_t>( "badAsicMask0", offsetof(CsPadConfigV5, badAsicMask0) ) ;
  confType.insert_native<uint32_t>( "badAsicMask1", offsetof(CsPadConfigV5, badAsicMask1) ) ;
  confType.insert_native<uint32_t>( "asicMask", offsetof(CsPadConfigV5, asicMask) ) ;
  confType.insert_native<uint32_t>( "quadMask", offsetof(CsPadConfigV5, quadMask) ) ;
  confType.insert_native<uint32_t>( "roiMask", offsetof(CsPadConfigV5, roiMask) ) ;
  confType.insert("quads", offsetof(CsPadConfigV5, quads), CsPadConfigV3QuadReg::native_type(), MaxQuadsPerSensor ) ;
  confType.insert("sections", offsetof(CsPadConfigV5, sections), sectArrType ) ;

  return confType ;
}

void
CsPadConfigV5::store( const XtcType& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  CsPadConfigV5 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
