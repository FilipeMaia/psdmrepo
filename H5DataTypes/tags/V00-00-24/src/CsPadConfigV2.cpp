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
#include "SITConfig/SITConfig.h"

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

  for ( int q = 0; q < CsPadConfigV2_Data::MaxQuadsPerSensor ; ++ q ) {
    m_data.roiMask[q] = data.roiMask(q);
  }
  
  for ( int q = 0; q < CsPadConfigV2_Data::MaxQuadsPerSensor ; ++ q ) {
    m_data.quads[q] = data.quads()[q];
  }

  // fill the list of active sections from ROI bits
  int ns = 0 ;
  for ( int q = 0; q < CsPadConfigV2_Data::MaxQuadsPerSensor ; ++ q ) {
    for ( int s = 0; s < CsPadConfigV2_Data::SectionsPerQuad ; ++ s ) {
      m_data.sections[q][s] = -1;
      if ( m_data.roiMask[q] & (1<<s) ) m_data.sections[q][s] = ns++;
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

  hsize_t sdims[2] = {CsPadConfigV2_Data::MaxQuadsPerSensor, CsPadConfigV2_Data::SectionsPerQuad};
  hdf5pp::Type baseSectType = hdf5pp::TypeTraits<int8_t>::native_type();
  hdf5pp::ArrayType sectArrType = hdf5pp::ArrayType::arrayType(baseSectType, 2, sdims);

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

  hdf5pp::ArrayType quadArrType = hdf5pp::ArrayType::arrayType(quadType, CsPadConfigV2_Data::MaxQuadsPerSensor);
  
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<CsPadConfigV2_Data>() ;
  confType.insert_native<uint32_t>( "concentratorVersion", offsetof(CsPadConfigV2_Data, concentratorVersion) ) ;
  confType.insert_native<uint32_t>( "runDelay", offsetof(CsPadConfigV2_Data, runDelay) ) ;
  confType.insert_native<uint32_t>( "eventCode", offsetof(CsPadConfigV2_Data, eventCode) ) ;
  confType.insert_native<uint32_t>( "inactiveRunMode", offsetof(CsPadConfigV2_Data, inactiveRunMode) ) ;
  confType.insert_native<uint32_t>( "activeRunMode", offsetof(CsPadConfigV2_Data, activeRunMode) ) ;
  confType.insert_native<uint32_t>( "testDataIndex", offsetof(CsPadConfigV2_Data, testDataIndex) ) ;
  confType.insert_native<uint32_t>( "payloadPerQuad", offsetof(CsPadConfigV2_Data, payloadPerQuad) ) ;
  confType.insert_native<uint32_t>( "badAsicMask0", offsetof(CsPadConfigV2_Data, badAsicMask0) ) ;
  confType.insert_native<uint32_t>( "badAsicMask1", offsetof(CsPadConfigV2_Data, badAsicMask1) ) ;
  confType.insert_native<uint32_t>( "asicMask", offsetof(CsPadConfigV2_Data, asicMask) ) ;
  confType.insert_native<uint32_t>( "quadMask", offsetof(CsPadConfigV2_Data, quadMask) ) ;
  confType.insert_native<uint32_t>( "roiMask", offsetof(CsPadConfigV2_Data, roiMask) ) ;
  confType.insert("quads", offsetof(CsPadConfigV2_Data, quads), quadArrType ) ;
  confType.insert("sections", offsetof(CsPadConfigV2_Data, sections), sectArrType ) ;

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
