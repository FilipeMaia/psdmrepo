#ifndef H5DATATYPES_EPIXASICCONFIGV1_H
#define H5DATATYPES_EPIXASICCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpixAsicConfigV1.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------


//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "hdf5pp/Group.h"
#include "pdsdata/psddl/epix.ddl.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Epix::AsicConfigV1
//
class EpixAsicConfigV1  {
public:

  typedef Pds::Epix::AsicConfigV1 XtcType;

  EpixAsicConfigV1() {}
  EpixAsicConfigV1(const XtcType& data);

  static hdf5pp::Type stored_type();
  static hdf5pp::Type native_type();

private:

  uint8_t monostPulser;
  uint8_t dummyTest;
  uint8_t dummyMask;
  uint16_t pulser;
  uint8_t pbit;
  uint8_t atest;
  uint8_t test;
  uint8_t sabTest;
  uint8_t hrTest;
  uint8_t digMon1;
  uint8_t digMon2;
  uint8_t pulserDac;
  uint8_t Dm1En;
  uint8_t Dm2En;
  uint8_t slvdSBit;
  uint8_t VRefDac;
  uint8_t TpsTComp;
  uint8_t TpsMux;
  uint8_t RoMonost;
  uint8_t TpsGr;
  uint8_t S2dGr;
  uint8_t PpOcbS2d;
  uint8_t Ocb;
  uint8_t Monost;
  uint8_t FastppEnable;
  uint8_t Preamp;
  uint8_t PixelCb;
  uint8_t S2dTComp;
  uint8_t FilterDac;
  uint8_t TC;
  uint8_t S2d;
  uint8_t S2dDacBias;
  uint8_t TpsTcDac;
  uint8_t TpsDac;
  uint8_t S2dTcDac;
  uint8_t S2dDac;
  uint8_t TestBe;
  uint8_t IsEn;
  uint8_t DelExec;
  uint8_t DelCckReg;
  uint16_t RowStartAddr;
  uint16_t RowStopAddr;
  uint8_t ColStartAddr;
  uint8_t ColStopAddr;
  uint16_t chipID;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_EPIXASICCONFIGV1_H
