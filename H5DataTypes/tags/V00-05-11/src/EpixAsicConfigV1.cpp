//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpixAsicConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/EpixAsicConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------

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

EpixAsicConfigV1::EpixAsicConfigV1(const XtcType& data)
  : monostPulser(data.monostPulser())
  , dummyTest(data.dummyTest())
  , dummyMask(data.dummyMask())
  , pulser(data.pulser())
  , pbit(data.pbit())
  , atest(data.atest())
  , test(data.test())
  , sabTest(data.sabTest())
  , hrTest(data.hrTest())
  , digMon1(data.digMon1())
  , digMon2(data.digMon2())
  , pulserDac(data.pulserDac())
  , Dm1En(data.Dm1En())
  , Dm2En(data.Dm2En())
  , slvdSBit(data.slvdSBit())
  , VRefDac(data.VRefDac())
  , TpsTComp(data.TpsTComp())
  , TpsMux(data.TpsMux())
  , RoMonost(data.RoMonost())
  , TpsGr(data.TpsGr())
  , S2dGr(data.S2dGr())
  , PpOcbS2d(data.PpOcbS2d())
  , Ocb(data.Ocb())
  , Monost(data.Monost())
  , FastppEnable(data.FastppEnable())
  , Preamp(data.Preamp())
  , PixelCb(data.PixelCb())
  , S2dTComp(data.S2dTComp())
  , FilterDac(data.FilterDac())
  , TC(data.TC())
  , S2d(data.S2d())
  , S2dDacBias(data.S2dDacBias())
  , TpsTcDac(data.TpsTcDac())
  , TpsDac(data.TpsDac())
  , S2dTcDac(data.S2dTcDac())
  , S2dDac(data.S2dDac())
  , TestBe(data.TestBe())
  , IsEn(data.IsEn())
  , DelExec(data.DelExec())
  , DelCckReg(data.DelCckReg())
  , RowStartAddr(data.RowStartAddr())
  , RowStopAddr(data.RowStopAddr())
  , ColStartAddr(data.ColStartAddr())
  , ColStopAddr(data.ColStopAddr())
  , chipID(data.chipID())
{
}

hdf5pp::Type
EpixAsicConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EpixAsicConfigV1::native_type()
{
  typedef EpixAsicConfigV1 DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("monostPulser", offsetof(DsType, monostPulser), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("dummyTest", offsetof(DsType, dummyTest), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("dummyMask", offsetof(DsType, dummyMask), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("pulser", offsetof(DsType, pulser), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("pbit", offsetof(DsType, pbit), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("atest", offsetof(DsType, atest), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("test", offsetof(DsType, test), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("sabTest", offsetof(DsType, sabTest), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("hrTest", offsetof(DsType, hrTest), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("digMon1", offsetof(DsType, digMon1), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("digMon2", offsetof(DsType, digMon2), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("pulserDac", offsetof(DsType, pulserDac), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("Dm1En", offsetof(DsType, Dm1En), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("Dm2En", offsetof(DsType, Dm2En), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("slvdSBit", offsetof(DsType, slvdSBit), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("VRefDac", offsetof(DsType, VRefDac), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("TpsTComp", offsetof(DsType, TpsTComp), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("TpsMux", offsetof(DsType, TpsMux), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("RoMonost", offsetof(DsType, RoMonost), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("TpsGr", offsetof(DsType, TpsGr), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("S2dGr", offsetof(DsType, S2dGr), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("PpOcbS2d", offsetof(DsType, PpOcbS2d), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("Ocb", offsetof(DsType, Ocb), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("Monost", offsetof(DsType, Monost), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("FastppEnable", offsetof(DsType, FastppEnable), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("Preamp", offsetof(DsType, Preamp), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("PixelCb", offsetof(DsType, PixelCb), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("S2dTComp", offsetof(DsType, S2dTComp), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("FilterDac", offsetof(DsType, FilterDac), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("TC", offsetof(DsType, TC), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("S2d", offsetof(DsType, S2d), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("S2dDacBias", offsetof(DsType, S2dDacBias), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("TpsTcDac", offsetof(DsType, TpsTcDac), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("TpsDac", offsetof(DsType, TpsDac), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("S2dTcDac", offsetof(DsType, S2dTcDac), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("S2dDac", offsetof(DsType, S2dDac), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("TestBe", offsetof(DsType, TestBe), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("IsEn", offsetof(DsType, IsEn), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("DelExec", offsetof(DsType, DelExec), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("DelCckReg", offsetof(DsType, DelCckReg), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("RowStartAddr", offsetof(DsType, RowStartAddr), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("RowStopAddr", offsetof(DsType, RowStopAddr), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("ColStartAddr", offsetof(DsType, ColStartAddr), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("ColStopAddr", offsetof(DsType, ColStopAddr), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("chipID", offsetof(DsType, chipID), hdf5pp::TypeTraits<uint16_t>::native_type());
  return type;
}

} // namespace H5DataTypes
