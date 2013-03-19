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

CsPadDigitalPotsCfg::CsPadDigitalPotsCfg(const Pds::CsPad::CsPadDigitalPotsCfg& o)
{
  std::copy(o.pots, o.pots+PotsPerQuad, this->pots);
}

hdf5pp::Type
CsPadDigitalPotsCfg::native_type()
{
  hdf5pp::CompoundType digitalPotsType = hdf5pp::CompoundType::compoundType<CsPadDigitalPotsCfg>() ;
  digitalPotsType.insert_native<uint8_t>( "pots", offsetof(CsPadDigitalPotsCfg, pots), PotsPerQuad );
  return digitalPotsType;
}


CsPadReadOnlyCfg::CsPadReadOnlyCfg(const Pds::CsPad::CsPadReadOnlyCfg& o)
  : shiftTest(o.shiftTest)
  , version(o.version)
{
}

hdf5pp::Type
CsPadReadOnlyCfg::native_type()
{
  hdf5pp::CompoundType readOnlyType = hdf5pp::CompoundType::compoundType<CsPadReadOnlyCfg>() ;
  readOnlyType.insert_native<uint32_t>( "shiftTest", offsetof(CsPadReadOnlyCfg, shiftTest) ) ;
  readOnlyType.insert_native<uint32_t>( "version", offsetof(CsPadReadOnlyCfg, version) ) ;
  return readOnlyType;
}


CsPadGainMapCfg::CsPadGainMapCfg(const Pds::CsPad::CsPadGainMapCfg& src)
{
  size_t size = ColumnsPerASIC * MaxRowsPerASIC;
  const uint16_t* srcmap = &src._gainMap[0][0];
  std::copy(srcmap, srcmap+size, &gainMap[0][0]);
}

hdf5pp::Type
CsPadGainMapCfg::native_type()
{
  hsize_t dims[2] = {ColumnsPerASIC, MaxRowsPerASIC};
  hdf5pp::Type baseMapType = hdf5pp::TypeTraits<uint16_t>::native_type();
  hdf5pp::ArrayType gainMapArrType = hdf5pp::ArrayType::arrayType(baseMapType, 2, dims);
  hdf5pp::CompoundType gainMapType = hdf5pp::CompoundType::compoundType<CsPadGainMapCfg>() ;
  gainMapType.insert( "gainMap", offsetof(CsPadGainMapCfg, gainMap), gainMapArrType );
  return gainMapType;
}


CsPadConfigV1QuadReg::CsPadConfigV1QuadReg(const Pds::CsPad::ConfigV1QuadReg& src)
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
CsPadConfigV1QuadReg::native_type()
{
  hdf5pp::CompoundType quadType = hdf5pp::CompoundType::compoundType<CsPadConfigV1QuadReg>() ;
  quadType.insert_native<uint32_t>( "shiftSelect", offsetof(CsPadConfigV1QuadReg, shiftSelect), TwoByTwosPerQuad ) ;
  quadType.insert_native<uint32_t>( "edgeSelect", offsetof(CsPadConfigV1QuadReg, edgeSelect), TwoByTwosPerQuad ) ;
  quadType.insert_native<uint32_t>( "readClkSet", offsetof(CsPadConfigV1QuadReg, readClkSet) ) ;
  quadType.insert_native<uint32_t>( "readClkHold", offsetof(CsPadConfigV1QuadReg, readClkHold) ) ;
  quadType.insert_native<uint32_t>( "dataMode", offsetof(CsPadConfigV1QuadReg, dataMode) ) ;
  quadType.insert_native<uint32_t>( "prstSel", offsetof(CsPadConfigV1QuadReg, prstSel) ) ;
  quadType.insert_native<uint32_t>( "acqDelay", offsetof(CsPadConfigV1QuadReg, acqDelay) ) ;
  quadType.insert_native<uint32_t>( "intTime", offsetof(CsPadConfigV1QuadReg, intTime) ) ;
  quadType.insert_native<uint32_t>( "digDelay", offsetof(CsPadConfigV1QuadReg, digDelay) ) ;
  quadType.insert_native<uint32_t>( "ampIdle", offsetof(CsPadConfigV1QuadReg, ampIdle) ) ;
  quadType.insert_native<uint32_t>( "injTotal", offsetof(CsPadConfigV1QuadReg, injTotal) ) ;
  quadType.insert_native<uint32_t>( "rowColShiftPer", offsetof(CsPadConfigV1QuadReg, rowColShiftPer) ) ;
  quadType.insert("readOnly", offsetof(CsPadConfigV1QuadReg, readOnly), CsPadReadOnlyCfg::native_type()) ;
  quadType.insert("digitalPots", offsetof(CsPadConfigV1QuadReg, digitalPots), CsPadDigitalPotsCfg::native_type()) ;
  quadType.insert("gainMap", offsetof(CsPadConfigV1QuadReg, gainMap), CsPadGainMapCfg::native_type()) ;
  return quadType;
}


CsPadConfigV1::CsPadConfigV1 ( const XtcType& data )
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
    quads[q] = data.quads()[q];
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
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<CsPadConfigV1>() ;
  confType.insert_native<uint32_t>( "concentratorVersion", offsetof(CsPadConfigV1, concentratorVersion) ) ;
  confType.insert_native<uint32_t>( "runDelay", offsetof(CsPadConfigV1, runDelay) ) ;
  confType.insert_native<uint32_t>( "eventCode", offsetof(CsPadConfigV1, eventCode) ) ;
  confType.insert_native<uint32_t>( "inactiveRunMode", offsetof(CsPadConfigV1, inactiveRunMode) ) ;
  confType.insert_native<uint32_t>( "activeRunMode", offsetof(CsPadConfigV1, activeRunMode) ) ;
  confType.insert_native<uint32_t>( "testDataIndex", offsetof(CsPadConfigV1, testDataIndex) ) ;
  confType.insert_native<uint32_t>( "payloadPerQuad", offsetof(CsPadConfigV1, payloadPerQuad) ) ;
  confType.insert_native<uint32_t>( "badAsicMask0", offsetof(CsPadConfigV1, badAsicMask0) ) ;
  confType.insert_native<uint32_t>( "badAsicMask1", offsetof(CsPadConfigV1, badAsicMask1) ) ;
  confType.insert_native<uint32_t>( "asicMask", offsetof(CsPadConfigV1, asicMask) ) ;
  confType.insert_native<uint32_t>( "quadMask", offsetof(CsPadConfigV1, quadMask) ) ;
  confType.insert("quads", offsetof(CsPadConfigV1, quads), CsPadConfigV1QuadReg::native_type(), MaxQuadsPerSensor ) ;

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
