//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2ConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/CsPad2x2ConfigV1.h"

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

CsPad2x2DigitalPotsCfg_Data&
CsPad2x2DigitalPotsCfg_Data::operator=(const Pds::CsPad2x2::CsPad2x2DigitalPotsCfg& o)
{
  std::copy(o.pots, o.pots+PotsPerQuad, this->pots);
  return *this;
}

CsPad2x2ReadOnlyCfg_Data&
CsPad2x2ReadOnlyCfg_Data::operator=(const Pds::CsPad2x2::CsPad2x2ReadOnlyCfg& o)
{
  this->shiftTest = o.shiftTest;
  this->version = o.version;
  return *this;
}

CsPad2x2GainMapCfg_Data&
CsPad2x2GainMapCfg_Data::operator=(const Pds::CsPad2x2::CsPad2x2GainMapCfg& src)
{
  size_t size = ColumnsPerASIC * MaxRowsPerASIC;
  const uint16_t* srcmap = &src._gainMap[0][0];
  std::copy(srcmap, srcmap+size, &gainMap[0][0]);
  return *this;
}

CsPad2x2ConfigV1QuadReg_Data&
CsPad2x2ConfigV1QuadReg_Data::operator=(const Pds::CsPad2x2::ConfigV1QuadReg& src)
{
  this->shiftSelect    = src.shiftSelect()[0];
  this->edgeSelect     = src.edgeSelect()[0];
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
  this->readOnly       = src.ro();
  this->digitalPots    = src.dp();
  this->gainMap        = *src.gm();

  return *this;
}

CsPad2x2ProtectionSystemThreshold_Data&
CsPad2x2ProtectionSystemThreshold_Data::operator=(const Pds::CsPad2x2::ProtectionSystemThreshold& o)
{
  this->adcThreshold = o.adcThreshold;
  this->pixelCountThreshold = o.pixelCountThreshold;
  return *this;
}


CsPad2x2ConfigV1::CsPad2x2ConfigV1 ( const XtcType& data )
{
  m_data.quad           = *data.quad();
  m_data.testDataIndex  = data.tdi();
  m_data.protectionThreshold = *data.protectionThreshold();
  m_data.protectionEnable = data.protectionEnable();
  m_data.inactiveRunMode = data.inactiveRunMode();
  m_data.activeRunMode  = data.activeRunMode();
  m_data.payloadSize    = data.payloadSize();
  m_data.badAsicMask    = data.badAsicMask();
  m_data.asicMask       = data.asicMask();
  m_data.roiMask        = data.roiMask();
  m_data.numAsicsRead   = data.numAsicsRead();
  m_data.numAsicsStored = data.numAsicsStored();
  m_data.concentratorVersion = data.concentratorVersion();
}

hdf5pp::Type
CsPad2x2ConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
CsPad2x2ConfigV1::native_type()
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

  hdf5pp::CompoundType quadType = hdf5pp::CompoundType::compoundType<CsPad2x2ConfigV1QuadReg_Data>() ;
  quadType.insert_native<uint32_t>( "shiftSelect", offsetof(CsPad2x2ConfigV1QuadReg_Data, shiftSelect) ) ;
  quadType.insert_native<uint32_t>( "edgeSelect", offsetof(CsPad2x2ConfigV1QuadReg_Data, edgeSelect) ) ;
  quadType.insert_native<uint32_t>( "readClkSet", offsetof(CsPad2x2ConfigV1QuadReg_Data, readClkSet) ) ;
  quadType.insert_native<uint32_t>( "readClkHold", offsetof(CsPad2x2ConfigV1QuadReg_Data, readClkHold) ) ;
  quadType.insert_native<uint32_t>( "dataMode", offsetof(CsPad2x2ConfigV1QuadReg_Data, dataMode) ) ;
  quadType.insert_native<uint32_t>( "prstSel", offsetof(CsPad2x2ConfigV1QuadReg_Data, prstSel) ) ;
  quadType.insert_native<uint32_t>( "acqDelay", offsetof(CsPad2x2ConfigV1QuadReg_Data, acqDelay) ) ;
  quadType.insert_native<uint32_t>( "intTime", offsetof(CsPad2x2ConfigV1QuadReg_Data, intTime) ) ;
  quadType.insert_native<uint32_t>( "digDelay", offsetof(CsPad2x2ConfigV1QuadReg_Data, digDelay) ) ;
  quadType.insert_native<uint32_t>( "ampIdle", offsetof(CsPad2x2ConfigV1QuadReg_Data, ampIdle) ) ;
  quadType.insert_native<uint32_t>( "injTotal", offsetof(CsPad2x2ConfigV1QuadReg_Data, injTotal) ) ;
  quadType.insert_native<uint32_t>( "rowColShiftPer", offsetof(CsPad2x2ConfigV1QuadReg_Data, rowColShiftPer) ) ;
  quadType.insert_native<uint32_t>( "ampReset", offsetof(CsPad2x2ConfigV1QuadReg_Data, ampReset) ) ;
  quadType.insert_native<uint32_t>( "digCount", offsetof(CsPad2x2ConfigV1QuadReg_Data, digCount) ) ;
  quadType.insert_native<uint32_t>( "digPeriod", offsetof(CsPad2x2ConfigV1QuadReg_Data, digPeriod) ) ;
  quadType.insert_native<uint32_t>( "PeltierEnable", offsetof(CsPad2x2ConfigV1QuadReg_Data, PeltierEnable) ) ;
  quadType.insert_native<uint32_t>( "kpConstant", offsetof(CsPad2x2ConfigV1QuadReg_Data, kpConstant) ) ;
  quadType.insert_native<uint32_t>( "kiConstant", offsetof(CsPad2x2ConfigV1QuadReg_Data, kiConstant) ) ;
  quadType.insert_native<uint32_t>( "kdConstant", offsetof(CsPad2x2ConfigV1QuadReg_Data, kdConstant) ) ;
  quadType.insert_native<uint32_t>( "humidThold", offsetof(CsPad2x2ConfigV1QuadReg_Data, humidThold) ) ;
  quadType.insert_native<uint32_t>( "setPoint", offsetof(CsPad2x2ConfigV1QuadReg_Data, setPoint) ) ;
  quadType.insert("readOnly", offsetof(CsPad2x2ConfigV1QuadReg_Data, readOnly), readOnlyType) ;
  quadType.insert("digitalPots", offsetof(CsPad2x2ConfigV1QuadReg_Data, digitalPots), digitalPotsType) ;
  quadType.insert("gainMap", offsetof(CsPad2x2ConfigV1QuadReg_Data, gainMap), gainMapType) ;

  hdf5pp::CompoundType protSysType = hdf5pp::CompoundType::compoundType<CsPad2x2ProtectionSystemThreshold_Data>() ;
  protSysType.insert_native<uint32_t>( "adcThreshold", offsetof(CsPad2x2ProtectionSystemThreshold_Data, adcThreshold) ) ;
  protSysType.insert_native<uint32_t>( "pixelCountThreshold", offsetof(CsPad2x2ProtectionSystemThreshold_Data, pixelCountThreshold) ) ;
  
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<CsPad2x2ConfigV1_Data>() ;
  confType.insert("quad", offsetof(CsPad2x2ConfigV1_Data, quad), quadType ) ;
  confType.insert_native<uint32_t>( "testDataIndex", offsetof(CsPad2x2ConfigV1_Data, testDataIndex) ) ;
  confType.insert("protectionThreshold", offsetof(CsPad2x2ConfigV1_Data, protectionThreshold), protSysType ) ;
  confType.insert_native<uint32_t>( "protectionEnable", offsetof(CsPad2x2ConfigV1_Data, protectionEnable) ) ;
  confType.insert_native<uint32_t>( "inactiveRunMode", offsetof(CsPad2x2ConfigV1_Data, inactiveRunMode) ) ;
  confType.insert_native<uint32_t>( "activeRunMode", offsetof(CsPad2x2ConfigV1_Data, activeRunMode) ) ;
  confType.insert_native<uint32_t>( "payloadSize", offsetof(CsPad2x2ConfigV1_Data, payloadSize) ) ;
  confType.insert_native<uint32_t>( "badAsicMask", offsetof(CsPad2x2ConfigV1_Data, badAsicMask) ) ;
  confType.insert_native<uint32_t>( "asicMask", offsetof(CsPad2x2ConfigV1_Data, asicMask) ) ;
  confType.insert_native<uint32_t>( "roiMask", offsetof(CsPad2x2ConfigV1_Data, roiMask) ) ;
  confType.insert_native<uint32_t>( "numAsicsRead", offsetof(CsPad2x2ConfigV1_Data, numAsicsRead) ) ;
  confType.insert_native<uint32_t>( "numAsicsStored", offsetof(CsPad2x2ConfigV1_Data, numAsicsStored) ) ;
  confType.insert_native<uint32_t>( "concentratorVersion", offsetof(CsPad2x2ConfigV1_Data, concentratorVersion) ) ;

  return confType ;
}

void
CsPad2x2ConfigV1::store( const XtcType& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  CsPad2x2ConfigV1 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
