//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/CsPadConfigV1.h"

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

CsPadDigitalPotsCfg_Data& 
CsPadDigitalPotsCfg_Data::operator=(const Pds::CsPad::CsPadDigitalPotsCfg& o)
{
  std::copy(o.pots, o.pots+PotsPerQuad, this->pots);
  return *this;
}

CsPadReadOnlyCfg_Data& 
CsPadReadOnlyCfg_Data::operator=(const Pds::CsPad::CsPadReadOnlyCfg& o)
{
  this->shiftTest = o.shiftTest;
  this->version = o.version;
  return *this;
}

CsPadGainMapCfg_Data& 
CsPadGainMapCfg_Data::operator=(const Pds::CsPad::CsPadGainMapCfg& src)
{
  size_t size = ColumnsPerASIC * MaxRowsPerASIC;
  const uint16_t* srcmap = &src._gainMap[0][0];
  std::copy(srcmap, srcmap+size, &gainMap[0][0]);
  return *this;
}

CsPadConfigV1QuadReg_Data& 
CsPadConfigV1QuadReg_Data::operator=(const Pds::CsPad::ConfigV1QuadReg& src)
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
  this->readOnly       = src.ro();
  this->digitalPots    = src.dp();
  this->gainMap        = *src.gm();
  
  return *this;
}

CsPadConfigV1::CsPadConfigV1 ( const XtcType& data )
{
  m_data.concentratorVersion = data.concentratorVersion();
  m_data.runDelay       = data.runDelay();
  m_data.eventCode      = data.eventCode();
  m_data.inactiveRunMode = data.inactiveRunMode();
  m_data.activeRunMode  = data.activeRunMode();
  m_data.testDataIndex  = data.tdi();
  m_data.payloadPerQuad = data.payloadSize();
  m_data.badAsicMask0   = data.badAsicMask0();
  m_data.badAsicMask1   = data.badAsicMask1();
  m_data.asicMask       = data.asicMask();
  m_data.quadMask       = data.quadMask();
  
  for ( int q = 0; q < CsPadConfigV1_Data::MaxQuadsPerSensor ; ++ q ) {
    m_data.quads[q] = data.quads()[q];
  }
}

hdf5pp::Type
CsPadConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
CsPadConfigV1::native_type()
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

  hdf5pp::ArrayType shiftSelectType = hdf5pp::ArrayType::arrayType<uint32_t>(CsPadConfigV1QuadReg_Data::TwoByTwosPerQuad) ;
  hdf5pp::ArrayType edgeSelectType = hdf5pp::ArrayType::arrayType<uint32_t>(CsPadConfigV1QuadReg_Data::TwoByTwosPerQuad) ;
  hdf5pp::CompoundType quadType = hdf5pp::CompoundType::compoundType<CsPadConfigV1QuadReg_Data>() ;
  quadType.insert( "shiftSelect", offsetof(CsPadConfigV1QuadReg_Data, shiftSelect), shiftSelectType ) ;
  quadType.insert( "edgeSelect", offsetof(CsPadConfigV1QuadReg_Data, edgeSelect), edgeSelectType ) ;
  quadType.insert_native<uint32_t>( "readClkSet", offsetof(CsPadConfigV1QuadReg_Data, readClkSet) ) ;
  quadType.insert_native<uint32_t>( "readClkHold", offsetof(CsPadConfigV1QuadReg_Data, readClkHold) ) ;
  quadType.insert_native<uint32_t>( "dataMode", offsetof(CsPadConfigV1QuadReg_Data, dataMode) ) ;
  quadType.insert_native<uint32_t>( "prstSel", offsetof(CsPadConfigV1QuadReg_Data, prstSel) ) ;
  quadType.insert_native<uint32_t>( "acqDelay", offsetof(CsPadConfigV1QuadReg_Data, acqDelay) ) ;
  quadType.insert_native<uint32_t>( "intTime", offsetof(CsPadConfigV1QuadReg_Data, intTime) ) ;
  quadType.insert_native<uint32_t>( "digDelay", offsetof(CsPadConfigV1QuadReg_Data, digDelay) ) ;
  quadType.insert_native<uint32_t>( "ampIdle", offsetof(CsPadConfigV1QuadReg_Data, ampIdle) ) ;
  quadType.insert_native<uint32_t>( "injTotal", offsetof(CsPadConfigV1QuadReg_Data, injTotal) ) ;
  quadType.insert_native<uint32_t>( "rowColShiftPer", offsetof(CsPadConfigV1QuadReg_Data, rowColShiftPer) ) ;
  quadType.insert("readOnly", offsetof(CsPadConfigV1QuadReg_Data, readOnly), readOnlyType) ;
  quadType.insert("digitalPots", offsetof(CsPadConfigV1QuadReg_Data, digitalPots), digitalPotsType) ;
  quadType.insert("gainMap", offsetof(CsPadConfigV1QuadReg_Data, gainMap), gainMapType) ;

  hdf5pp::ArrayType quadArrType = hdf5pp::ArrayType::arrayType(quadType, CsPadConfigV1_Data::MaxQuadsPerSensor);
  
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<CsPadConfigV1_Data>() ;
  confType.insert_native<uint32_t>( "concentratorVersion", offsetof(CsPadConfigV1_Data, concentratorVersion) ) ;
  confType.insert_native<uint32_t>( "runDelay", offsetof(CsPadConfigV1_Data, runDelay) ) ;
  confType.insert_native<uint32_t>( "eventCode", offsetof(CsPadConfigV1_Data, eventCode) ) ;
  confType.insert_native<uint32_t>( "inactiveRunMode", offsetof(CsPadConfigV1_Data, inactiveRunMode) ) ;
  confType.insert_native<uint32_t>( "activeRunMode", offsetof(CsPadConfigV1_Data, activeRunMode) ) ;
  confType.insert_native<uint32_t>( "testDataIndex", offsetof(CsPadConfigV1_Data, testDataIndex) ) ;
  confType.insert_native<uint32_t>( "payloadPerQuad", offsetof(CsPadConfigV1_Data, payloadPerQuad) ) ;
  confType.insert_native<uint32_t>( "badAsicMask0", offsetof(CsPadConfigV1_Data, badAsicMask0) ) ;
  confType.insert_native<uint32_t>( "badAsicMask1", offsetof(CsPadConfigV1_Data, badAsicMask1) ) ;
  confType.insert_native<uint32_t>( "asicMask", offsetof(CsPadConfigV1_Data, asicMask) ) ;
  confType.insert_native<uint32_t>( "quadMask", offsetof(CsPadConfigV1_Data, quadMask) ) ;
  confType.insert("quads", offsetof(CsPadConfigV1_Data, quads), quadArrType ) ;

  return confType ;
}

void
CsPadConfigV1::store( const XtcType& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  CsPadConfigV1 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
