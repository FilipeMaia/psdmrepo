#ifndef H5DATATYPES_CSPADCONFIGV1_H
#define H5DATATYPES_CSPADCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadConfigV1.
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
#include "hdf5pp/Group.h"
#include "pdsdata/cspad/ConfigV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {
  
  
//
// Helper type for Pds::CsPad::CsPadDigitalPotsCfg
//
struct CsPadDigitalPotsCfg  {
  CsPadDigitalPotsCfg() {}
  CsPadDigitalPotsCfg(const Pds::CsPad::CsPadDigitalPotsCfg& o);

  static hdf5pp::Type native_type() ;

private:
  enum { PotsPerQuad = Pds::CsPad::PotsPerQuad };
  uint8_t         pots[PotsPerQuad];
};

//
// Helper type for Pds::CsPad::CsPadReadOnlyCfg
//
struct CsPadReadOnlyCfg  {
  CsPadReadOnlyCfg() {}
  CsPadReadOnlyCfg(const Pds::CsPad::CsPadReadOnlyCfg& o);

  static hdf5pp::Type native_type() ;

private:
  uint32_t        shiftTest;
  uint32_t        version;
};

//
// Helper type for Pds::CsPad::CsPadGainMapCfg
//
struct CsPadGainMapCfg  {
  CsPadGainMapCfg() {}
  CsPadGainMapCfg(const Pds::CsPad::CsPadGainMapCfg& o);

  static hdf5pp::Type native_type() ;

private:
  enum { ColumnsPerASIC = Pds::CsPad::ColumnsPerASIC };
  enum { MaxRowsPerASIC = Pds::CsPad::MaxRowsPerASIC };
  uint16_t gainMap[ColumnsPerASIC][MaxRowsPerASIC];
};

//
// Helper type for Pds::CsPad::ConfigV1QuadReg
//
struct CsPadConfigV1QuadReg  {
  CsPadConfigV1QuadReg() {}
  CsPadConfigV1QuadReg(const Pds::CsPad::ConfigV1QuadReg& o);

  static hdf5pp::Type native_type() ;

private:
  enum { TwoByTwosPerQuad = Pds::CsPad::TwoByTwosPerQuad};
  uint32_t                  shiftSelect[TwoByTwosPerQuad];
  uint32_t                  edgeSelect[TwoByTwosPerQuad];
  uint32_t                  readClkSet;
  uint32_t                  readClkHold;
  uint32_t                  dataMode;
  uint32_t                  prstSel;
  uint32_t                  acqDelay;
  uint32_t                  intTime;
  uint32_t                  digDelay;
  uint32_t                  ampIdle;
  uint32_t                  injTotal;
  uint32_t                  rowColShiftPer;
  CsPadReadOnlyCfg          readOnly;
  CsPadDigitalPotsCfg       digitalPots;
  CsPadGainMapCfg           gainMap;
};

//
// Helper type for Pds::CsPad::ConfigV1
//
class CsPadConfigV1  {
public:

  typedef Pds::CsPad::ConfigV1 XtcType ;

  CsPadConfigV1 () {}
  CsPadConfigV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

private:

  enum { MaxQuadsPerSensor = Pds::CsPad::MaxQuadsPerSensor };
  uint32_t          concentratorVersion;
  uint32_t          runDelay;
  uint32_t          eventCode;
  uint32_t          inactiveRunMode;
  uint32_t          activeRunMode;
  uint32_t          testDataIndex;
  uint32_t          payloadPerQuad;
  uint32_t          badAsicMask0;
  uint32_t          badAsicMask1;
  uint32_t          asicMask;
  uint32_t          quadMask;
  CsPadConfigV1QuadReg quads[MaxQuadsPerSensor];

};

} // namespace H5DataTypes

#endif // H5DATATYPES_CSPADCONFIGV1_H
