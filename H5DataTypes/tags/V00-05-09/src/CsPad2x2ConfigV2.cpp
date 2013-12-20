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

CsPad2x2ConfigV2QuadReg::CsPad2x2ConfigV2QuadReg(const Pds::CsPad2x2::ConfigV2QuadReg& src)
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
  , gainMap(src.gm())
{
}

hdf5pp::Type
CsPad2x2ConfigV2QuadReg::native_type()
{
  hdf5pp::CompoundType quadType = hdf5pp::CompoundType::compoundType<CsPad2x2ConfigV2QuadReg>() ;
  quadType.insert_native<uint32_t>( "shiftSelect", offsetof(CsPad2x2ConfigV2QuadReg, shiftSelect) ) ;
  quadType.insert_native<uint32_t>( "edgeSelect", offsetof(CsPad2x2ConfigV2QuadReg, edgeSelect) ) ;
  quadType.insert_native<uint32_t>( "readClkSet", offsetof(CsPad2x2ConfigV2QuadReg, readClkSet) ) ;
  quadType.insert_native<uint32_t>( "readClkHold", offsetof(CsPad2x2ConfigV2QuadReg, readClkHold) ) ;
  quadType.insert_native<uint32_t>( "dataMode", offsetof(CsPad2x2ConfigV2QuadReg, dataMode) ) ;
  quadType.insert_native<uint32_t>( "prstSel", offsetof(CsPad2x2ConfigV2QuadReg, prstSel) ) ;
  quadType.insert_native<uint32_t>( "acqDelay", offsetof(CsPad2x2ConfigV2QuadReg, acqDelay) ) ;
  quadType.insert_native<uint32_t>( "intTime", offsetof(CsPad2x2ConfigV2QuadReg, intTime) ) ;
  quadType.insert_native<uint32_t>( "digDelay", offsetof(CsPad2x2ConfigV2QuadReg, digDelay) ) ;
  quadType.insert_native<uint32_t>( "ampIdle", offsetof(CsPad2x2ConfigV2QuadReg, ampIdle) ) ;
  quadType.insert_native<uint32_t>( "injTotal", offsetof(CsPad2x2ConfigV2QuadReg, injTotal) ) ;
  quadType.insert_native<uint32_t>( "rowColShiftPer", offsetof(CsPad2x2ConfigV2QuadReg, rowColShiftPer) ) ;
  quadType.insert_native<uint32_t>( "ampReset", offsetof(CsPad2x2ConfigV2QuadReg, ampReset) ) ;
  quadType.insert_native<uint32_t>( "digCount", offsetof(CsPad2x2ConfigV2QuadReg, digCount) ) ;
  quadType.insert_native<uint32_t>( "digPeriod", offsetof(CsPad2x2ConfigV2QuadReg, digPeriod) ) ;
  quadType.insert_native<uint32_t>( "PeltierEnable", offsetof(CsPad2x2ConfigV2QuadReg, PeltierEnable) ) ;
  quadType.insert_native<uint32_t>( "kpConstant", offsetof(CsPad2x2ConfigV2QuadReg, kpConstant) ) ;
  quadType.insert_native<uint32_t>( "kiConstant", offsetof(CsPad2x2ConfigV2QuadReg, kiConstant) ) ;
  quadType.insert_native<uint32_t>( "kdConstant", offsetof(CsPad2x2ConfigV2QuadReg, kdConstant) ) ;
  quadType.insert_native<uint32_t>( "humidThold", offsetof(CsPad2x2ConfigV2QuadReg, humidThold) ) ;
  quadType.insert_native<uint32_t>( "setPoint", offsetof(CsPad2x2ConfigV2QuadReg, setPoint) ) ;
  quadType.insert_native<uint32_t>( "biasTuning", offsetof(CsPad2x2ConfigV2QuadReg, biasTuning) ) ;
  quadType.insert_native<uint32_t>( "pdpmndnmBalance", offsetof(CsPad2x2ConfigV2QuadReg, pdpmndnmBalance) ) ;
  quadType.insert("readOnly", offsetof(CsPad2x2ConfigV2QuadReg, readOnly), CsPad2x2ReadOnlyCfg::native_type()) ;
  quadType.insert("digitalPots", offsetof(CsPad2x2ConfigV2QuadReg, digitalPots), CsPad2x2DigitalPotsCfg::native_type()) ;
  quadType.insert("gainMap", offsetof(CsPad2x2ConfigV2QuadReg, gainMap), CsPad2x2GainMapCfg::native_type()) ;
  return quadType;
}


CsPad2x2ConfigV2::CsPad2x2ConfigV2 ( const XtcType& data )
  : quad(data.quad())
  , testDataIndex(data.tdi())
  , protectionThreshold(data.protectionThreshold())
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
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<CsPad2x2ConfigV2>() ;
  confType.insert("quad", offsetof(CsPad2x2ConfigV2, quad), CsPad2x2ConfigV2QuadReg::native_type() ) ;
  confType.insert_native<uint32_t>( "testDataIndex", offsetof(CsPad2x2ConfigV2, testDataIndex) ) ;
  confType.insert("protectionThreshold", offsetof(CsPad2x2ConfigV2, protectionThreshold), CsPad2x2ProtectionSystemThreshold::native_type() ) ;
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
