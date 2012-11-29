//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadConfigV4...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/CsPadConfigV4.h"

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

CsPadConfigV2QuadReg_Data& 
CsPadConfigV2QuadReg_Data::operator=(const Pds::CsPad::ConfigV2QuadReg& src)
{
  const uint32_t* p = src.shiftSelect();
  std::copy(p, p+TwoByTwosPerQuad, this->shiftSelect);
  p = src.edgeSelect();
  std::copy(p, p+TwoByTwosPerQuad, this->edgeSelect);
  this->readClkSet     = src.readClkSet();
  this->readClkHold    = src.readClkHold();
  this->dataMode       = src.dataMode();
  this->prstSel        = src.prstSel();
  this->acqDelay       = src.acqDelay();
  this->intTime        = src.intTime();
  this->digDelay       = src.digDelay();
  this->ampIdle        = src.ampIdle();
  this->injTotal       = src.injTotal();
  this->rowColShiftPer = src.rowColShiftPer();
  this->ampReset       = src.ampReset();
  this->digCount       = src.digCount();
  this->digPeriod      = src.digPeriod();
  this->readOnly       = src.ro();
  this->digitalPots    = src.dp();
  this->gainMap        = *src.gm();
  
  return *this;
}

CsPadConfigV4::CsPadConfigV4 ( const XtcType& data )
  :  concentratorVersion(data.concentratorVersion())
  ,  runDelay(data.runDelay())
  ,  eventCode(data.eventCode())
  ,  protectionEnable(data.protectionEnable())
  ,  inactiveRunMode(data.inactiveRunMode())
  ,  activeRunMode(data.activeRunMode())
  ,  testDataIndex(data.tdi())
  ,  payloadPerQuad(data.payloadSize())
  ,  badAsicMask0(data.badAsicMask0())
  ,  badAsicMask1(data.badAsicMask1())
  ,  asicMask(data.asicMask())
  ,  quadMask(data.quadMask())
{

  for ( int q = 0; q < CsPadConfigV4::MaxQuadsPerSensor ; ++ q ) {
    roiMask[q] = data.roiMask(q);
  }
  
  for ( int q = 0; q < CsPadConfigV4::MaxQuadsPerSensor ; ++ q ) {
    quads[q] = data.quads()[q];
  }

  for ( int q = 0; q < CsPadConfigV4::MaxQuadsPerSensor ; ++ q ) {
    protectionThresholds[q] = data.protectionThresholds()[q];
  }

  // fill the list of active sections from ROI bits
  int ns = 0 ;
  for ( int q = 0; q < CsPadConfigV4::MaxQuadsPerSensor ; ++ q ) {
    for ( int s = 0; s < CsPadConfigV4::SectionsPerQuad ; ++ s ) {
      sections[q][s] = -1;
      if ( roiMask[q] & (1<<s) ) sections[q][s] = ns++;
    }
  }
  
}

hdf5pp::Type
CsPadConfigV4::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
CsPadConfigV4::native_type()
{
  hdf5pp::ArrayType potsType = hdf5pp::ArrayType::arrayType<uint8_t>(CsPadDigitalPotsCfg_Data::PotsPerQuad) ;
  hdf5pp::CompoundType digitalPotsType = hdf5pp::CompoundType::compoundType<CsPadDigitalPotsCfg_Data>() ;
  digitalPotsType.insert( "pots", offsetof(CsPadDigitalPotsCfg_Data, pots), potsType );

  hdf5pp::CompoundType readOnlyType = hdf5pp::CompoundType::compoundType<CsPadReadOnlyCfg_Data>() ;
  readOnlyType.insert_native<uint32_t>( "shiftTest", offsetof(CsPadReadOnlyCfg_Data, shiftTest) ) ;
  readOnlyType.insert_native<uint32_t>( "version", offsetof(CsPadReadOnlyCfg_Data, version) ) ;
  
  hsize_t dims[2] = {CsPadGainMapCfg_Data::ColumnsPerASIC, CsPadGainMapCfg_Data::MaxRowsPerASIC};
  hdf5pp::Type baseMapType = hdf5pp::TypeTraits<uint16_t>::native_type();
  hdf5pp::ArrayType gainMapArrType = hdf5pp::ArrayType::arrayType(baseMapType, 2, dims);
  hdf5pp::CompoundType gainMapType = hdf5pp::CompoundType::compoundType<CsPadGainMapCfg_Data>() ;
  gainMapType.insert( "gainMap", offsetof(CsPadGainMapCfg_Data, gainMap), gainMapArrType );

  hsize_t sdims[2] = {CsPadConfigV4::MaxQuadsPerSensor, CsPadConfigV4::SectionsPerQuad};
  hdf5pp::Type baseSectType = hdf5pp::TypeTraits<int8_t>::native_type();
  hdf5pp::ArrayType sectArrType = hdf5pp::ArrayType::arrayType(baseSectType, 2, sdims);

  hdf5pp::ArrayType shiftSelectType = hdf5pp::ArrayType::arrayType<uint32_t>(CsPadConfigV2QuadReg_Data::TwoByTwosPerQuad) ;
  hdf5pp::ArrayType edgeSelectType = hdf5pp::ArrayType::arrayType<uint32_t>(CsPadConfigV2QuadReg_Data::TwoByTwosPerQuad) ;
  hdf5pp::CompoundType quadType = hdf5pp::CompoundType::compoundType<CsPadConfigV2QuadReg_Data>() ;
  quadType.insert( "shiftSelect", offsetof(CsPadConfigV2QuadReg_Data, shiftSelect), shiftSelectType ) ;
  quadType.insert( "edgeSelect", offsetof(CsPadConfigV2QuadReg_Data, edgeSelect), edgeSelectType ) ;
  quadType.insert_native<uint32_t>( "readClkSet", offsetof(CsPadConfigV2QuadReg_Data, readClkSet) ) ;
  quadType.insert_native<uint32_t>( "readClkHold", offsetof(CsPadConfigV2QuadReg_Data, readClkHold) ) ;
  quadType.insert_native<uint32_t>( "dataMode", offsetof(CsPadConfigV2QuadReg_Data, dataMode) ) ;
  quadType.insert_native<uint32_t>( "prstSel", offsetof(CsPadConfigV2QuadReg_Data, prstSel) ) ;
  quadType.insert_native<uint32_t>( "acqDelay", offsetof(CsPadConfigV2QuadReg_Data, acqDelay) ) ;
  quadType.insert_native<uint32_t>( "intTime", offsetof(CsPadConfigV2QuadReg_Data, intTime) ) ;
  quadType.insert_native<uint32_t>( "digDelay", offsetof(CsPadConfigV2QuadReg_Data, digDelay) ) ;
  quadType.insert_native<uint32_t>( "ampIdle", offsetof(CsPadConfigV2QuadReg_Data, ampIdle) ) ;
  quadType.insert_native<uint32_t>( "injTotal", offsetof(CsPadConfigV2QuadReg_Data, injTotal) ) ;
  quadType.insert_native<uint32_t>( "rowColShiftPer", offsetof(CsPadConfigV2QuadReg_Data, rowColShiftPer) ) ;
  quadType.insert_native<uint32_t>( "ampReset", offsetof(CsPadConfigV2QuadReg_Data, ampReset) ) ;
  quadType.insert_native<uint32_t>( "digCount", offsetof(CsPadConfigV2QuadReg_Data, digCount) ) ;
  quadType.insert_native<uint32_t>( "digPeriod", offsetof(CsPadConfigV2QuadReg_Data, digPeriod) ) ;
  quadType.insert("readOnly", offsetof(CsPadConfigV2QuadReg_Data, readOnly), readOnlyType) ;
  quadType.insert("digitalPots", offsetof(CsPadConfigV2QuadReg_Data, digitalPots), digitalPotsType) ;
  quadType.insert("gainMap", offsetof(CsPadConfigV2QuadReg_Data, gainMap), gainMapType) ;

  hdf5pp::ArrayType quadArrType = hdf5pp::ArrayType::arrayType(quadType, CsPadConfigV4::MaxQuadsPerSensor);

  hdf5pp::CompoundType protSysType = hdf5pp::CompoundType::compoundType<CsPadProtectionSystemThreshold_Data>() ;
  protSysType.insert_native<uint32_t>( "adcThreshold", offsetof(CsPadProtectionSystemThreshold_Data, adcThreshold) ) ;
  protSysType.insert_native<uint32_t>( "pixelCountThreshold", offsetof(CsPadProtectionSystemThreshold_Data, pixelCountThreshold) ) ;
  hdf5pp::ArrayType protArrType = hdf5pp::ArrayType::arrayType(protSysType, CsPadConfigV4::MaxQuadsPerSensor);
  
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<CsPadConfigV4>() ;
  confType.insert_native<uint32_t>( "concentratorVersion", offsetof(CsPadConfigV4, concentratorVersion) ) ;
  confType.insert_native<uint32_t>( "runDelay", offsetof(CsPadConfigV4, runDelay) ) ;
  confType.insert_native<uint32_t>( "eventCode", offsetof(CsPadConfigV4, eventCode) ) ;
  confType.insert("protectionThresholds", offsetof(CsPadConfigV4, protectionThresholds), protArrType ) ;
  confType.insert_native<uint32_t>( "protectionEnable", offsetof(CsPadConfigV4, protectionEnable) ) ;
  confType.insert_native<uint32_t>( "inactiveRunMode", offsetof(CsPadConfigV4, inactiveRunMode) ) ;
  confType.insert_native<uint32_t>( "activeRunMode", offsetof(CsPadConfigV4, activeRunMode) ) ;
  confType.insert_native<uint32_t>( "testDataIndex", offsetof(CsPadConfigV4, testDataIndex) ) ;
  confType.insert_native<uint32_t>( "payloadPerQuad", offsetof(CsPadConfigV4, payloadPerQuad) ) ;
  confType.insert_native<uint32_t>( "badAsicMask0", offsetof(CsPadConfigV4, badAsicMask0) ) ;
  confType.insert_native<uint32_t>( "badAsicMask1", offsetof(CsPadConfigV4, badAsicMask1) ) ;
  confType.insert_native<uint32_t>( "asicMask", offsetof(CsPadConfigV4, asicMask) ) ;
  confType.insert_native<uint32_t>( "quadMask", offsetof(CsPadConfigV4, quadMask) ) ;
  confType.insert_native<uint32_t>( "roiMask", offsetof(CsPadConfigV4, roiMask) ) ;
  confType.insert("quads", offsetof(CsPadConfigV4, quads), quadArrType ) ;
  confType.insert("sections", offsetof(CsPadConfigV4, sections), sectArrType ) ;

  return confType ;
}

void
CsPadConfigV4::store( const XtcType& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  CsPadConfigV4 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
