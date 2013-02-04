//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2ConfigV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/CsPad2x2ConfigV2.h"

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

CsPad2x2ConfigV2QuadReg_Data::CsPad2x2ConfigV2QuadReg_Data(const Pds::CsPad2x2::ConfigV2QuadReg& src)
  : shiftSelect(src.shiftSelect())
  , edgeSelect(src.edgeSelect())
  , readClkSet(src.readClkSet())
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
  , PeltierEnable(src.PeltierEnable())
  , kpConstant(src.kpConstant())
  , kiConstant(src.kiConstant())
  , kdConstant(src.kdConstant())
  , humidThold(src.humidThold())
  , setPoint(src.setPoint())
  , biasTuning(src.biasTuning())
  , pdpmndnmBalance(src.pdpmndnmBalance())
  , readOnly(src.ro())
  , digitalPots(src.dp())
  , gainMap(*src.gm())
{
}

CsPad2x2ConfigV2::CsPad2x2ConfigV2 ( const XtcType& data )
  : quad(*data.quad())
  , testDataIndex(data.tdi())
  , protectionThreshold(*data.protectionThreshold())
  , protectionEnable(data.protectionEnable())
  , inactiveRunMode(data.inactiveRunMode())
  , activeRunMode(data.activeRunMode())
  , runTriggerDelay(data.runTriggerDelay())
  , payloadSize(data.payloadSize())
  , badAsicMask(data.badAsicMask())
  , asicMask(data.asicMask())
  , roiMask(data.roiMask())
  , numAsicsRead(data.numAsicsRead())
  , numAsicsStored(data.numAsicsStored())
  , concentratorVersion(data.concentratorVersion())
{
}

hdf5pp::Type
CsPad2x2ConfigV2::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
CsPad2x2ConfigV2::native_type()
{
  hdf5pp::ArrayType potsType = hdf5pp::ArrayType::arrayType<uint8_t>(CsPad2x2DigitalPotsCfg_Data::PotsPerQuad) ;
  hdf5pp::CompoundType digitalPotsType = hdf5pp::CompoundType::compoundType<CsPad2x2DigitalPotsCfg_Data>() ;
  digitalPotsType.insert( "pots", offsetof(CsPad2x2DigitalPotsCfg_Data, pots), potsType );

  hdf5pp::CompoundType readOnlyType = hdf5pp::CompoundType::compoundType<CsPad2x2ReadOnlyCfg_Data>() ;
  readOnlyType.insert_native<uint32_t>( "shiftTest", offsetof(CsPad2x2ReadOnlyCfg_Data, shiftTest) ) ;
  readOnlyType.insert_native<uint32_t>( "version", offsetof(CsPad2x2ReadOnlyCfg_Data, version) ) ;
  
  hsize_t dims[2] = {CsPad2x2GainMapCfg_Data::ColumnsPerASIC, CsPad2x2GainMapCfg_Data::MaxRowsPerASIC};
  hdf5pp::Type baseMapType = hdf5pp::TypeTraits<uint16_t>::native_type();
  hdf5pp::ArrayType gainMapArrType = hdf5pp::ArrayType::arrayType(baseMapType, 2, dims);
  hdf5pp::CompoundType gainMapType = hdf5pp::CompoundType::compoundType<CsPad2x2GainMapCfg_Data>() ;
  gainMapType.insert( "gainMap", offsetof(CsPad2x2GainMapCfg_Data, gainMap), gainMapArrType );

  hdf5pp::CompoundType quadType = hdf5pp::CompoundType::compoundType<CsPad2x2ConfigV2QuadReg_Data>() ;
  quadType.insert_native<uint32_t>( "shiftSelect", offsetof(CsPad2x2ConfigV2QuadReg_Data, shiftSelect) ) ;
  quadType.insert_native<uint32_t>( "edgeSelect", offsetof(CsPad2x2ConfigV2QuadReg_Data, edgeSelect) ) ;
  quadType.insert_native<uint32_t>( "readClkSet", offsetof(CsPad2x2ConfigV2QuadReg_Data, readClkSet) ) ;
  quadType.insert_native<uint32_t>( "readClkHold", offsetof(CsPad2x2ConfigV2QuadReg_Data, readClkHold) ) ;
  quadType.insert_native<uint32_t>( "dataMode", offsetof(CsPad2x2ConfigV2QuadReg_Data, dataMode) ) ;
  quadType.insert_native<uint32_t>( "prstSel", offsetof(CsPad2x2ConfigV2QuadReg_Data, prstSel) ) ;
  quadType.insert_native<uint32_t>( "acqDelay", offsetof(CsPad2x2ConfigV2QuadReg_Data, acqDelay) ) ;
  quadType.insert_native<uint32_t>( "intTime", offsetof(CsPad2x2ConfigV2QuadReg_Data, intTime) ) ;
  quadType.insert_native<uint32_t>( "digDelay", offsetof(CsPad2x2ConfigV2QuadReg_Data, digDelay) ) ;
  quadType.insert_native<uint32_t>( "ampIdle", offsetof(CsPad2x2ConfigV2QuadReg_Data, ampIdle) ) ;
  quadType.insert_native<uint32_t>( "injTotal", offsetof(CsPad2x2ConfigV2QuadReg_Data, injTotal) ) ;
  quadType.insert_native<uint32_t>( "rowColShiftPer", offsetof(CsPad2x2ConfigV2QuadReg_Data, rowColShiftPer) ) ;
  quadType.insert_native<uint32_t>( "ampReset", offsetof(CsPad2x2ConfigV2QuadReg_Data, ampReset) ) ;
  quadType.insert_native<uint32_t>( "digCount", offsetof(CsPad2x2ConfigV2QuadReg_Data, digCount) ) ;
  quadType.insert_native<uint32_t>( "digPeriod", offsetof(CsPad2x2ConfigV2QuadReg_Data, digPeriod) ) ;
  quadType.insert_native<uint32_t>( "PeltierEnable", offsetof(CsPad2x2ConfigV2QuadReg_Data, PeltierEnable) ) ;
  quadType.insert_native<uint32_t>( "kpConstant", offsetof(CsPad2x2ConfigV2QuadReg_Data, kpConstant) ) ;
  quadType.insert_native<uint32_t>( "kiConstant", offsetof(CsPad2x2ConfigV2QuadReg_Data, kiConstant) ) ;
  quadType.insert_native<uint32_t>( "kdConstant", offsetof(CsPad2x2ConfigV2QuadReg_Data, kdConstant) ) ;
  quadType.insert_native<uint32_t>( "humidThold", offsetof(CsPad2x2ConfigV2QuadReg_Data, humidThold) ) ;
  quadType.insert_native<uint32_t>( "setPoint", offsetof(CsPad2x2ConfigV2QuadReg_Data, setPoint) ) ;
  quadType.insert_native<uint32_t>( "setPoint", offsetof(CsPad2x2ConfigV2QuadReg_Data, setPoint) ) ;
  quadType.insert_native<uint32_t>( "biasTuning", offsetof(CsPad2x2ConfigV2QuadReg_Data, biasTuning) ) ;
  quadType.insert_native<uint32_t>( "pdpmndnmBalance", offsetof(CsPad2x2ConfigV2QuadReg_Data, pdpmndnmBalance) ) ;
  quadType.insert("readOnly", offsetof(CsPad2x2ConfigV2QuadReg_Data, readOnly), readOnlyType) ;
  quadType.insert("digitalPots", offsetof(CsPad2x2ConfigV2QuadReg_Data, digitalPots), digitalPotsType) ;
  quadType.insert("gainMap", offsetof(CsPad2x2ConfigV2QuadReg_Data, gainMap), gainMapType) ;

  hdf5pp::CompoundType protSysType = hdf5pp::CompoundType::compoundType<CsPad2x2ProtectionSystemThreshold_Data>() ;
  protSysType.insert_native<uint32_t>( "adcThreshold", offsetof(CsPad2x2ProtectionSystemThreshold_Data, adcThreshold) ) ;
  protSysType.insert_native<uint32_t>( "pixelCountThreshold", offsetof(CsPad2x2ProtectionSystemThreshold_Data, pixelCountThreshold) ) ;
  
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<CsPad2x2ConfigV2>() ;
  confType.insert("quad", offsetof(CsPad2x2ConfigV2, quad), quadType ) ;
  confType.insert_native<uint32_t>( "testDataIndex", offsetof(CsPad2x2ConfigV2, testDataIndex) ) ;
  confType.insert("protectionThreshold", offsetof(CsPad2x2ConfigV2, protectionThreshold), protSysType ) ;
  confType.insert_native<uint32_t>( "protectionEnable", offsetof(CsPad2x2ConfigV2, protectionEnable) ) ;
  confType.insert_native<uint32_t>( "inactiveRunMode", offsetof(CsPad2x2ConfigV2, inactiveRunMode) ) ;
  confType.insert_native<uint32_t>( "activeRunMode", offsetof(CsPad2x2ConfigV2, activeRunMode) ) ;
  confType.insert_native<uint32_t>( "runTriggerDelay", offsetof(CsPad2x2ConfigV2, runTriggerDelay) ) ;
  confType.insert_native<uint32_t>( "payloadSize", offsetof(CsPad2x2ConfigV2, payloadSize) ) ;
  confType.insert_native<uint32_t>( "badAsicMask", offsetof(CsPad2x2ConfigV2, badAsicMask) ) ;
  confType.insert_native<uint32_t>( "asicMask", offsetof(CsPad2x2ConfigV2, asicMask) ) ;
  confType.insert_native<uint32_t>( "roiMask", offsetof(CsPad2x2ConfigV2, roiMask) ) ;
  confType.insert_native<uint32_t>( "numAsicsRead", offsetof(CsPad2x2ConfigV2, numAsicsRead) ) ;
  confType.insert_native<uint32_t>( "numAsicsStored", offsetof(CsPad2x2ConfigV2, numAsicsStored) ) ;
  confType.insert_native<uint32_t>( "concentratorVersion", offsetof(CsPad2x2ConfigV2, concentratorVersion) ) ;

  return confType ;
}

void
CsPad2x2ConfigV2::store( const XtcType& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  CsPad2x2ConfigV2 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
