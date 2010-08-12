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
struct CsPadDigitalPotsCfg_Data  {
  enum { PotsPerQuad = Pds::CsPad::PotsPerQuad };
  uint8_t         pots[PotsPerQuad];
  
  CsPadDigitalPotsCfg_Data& operator=(const Pds::CsPad::CsPadDigitalPotsCfg& o);
};

//
// Helper type for Pds::CsPad::CsPadReadOnlyCfg
//
struct CsPadReadOnlyCfg_Data  {
  uint32_t        shiftTest;
  uint32_t        version;

  CsPadReadOnlyCfg_Data& operator=(const Pds::CsPad::CsPadReadOnlyCfg& o);
};

//
// Helper type for Pds::CsPad::CsPadGainMapCfg
//
struct CsPadGainMapCfg_Data  {
  enum { ColumnsPerASIC = Pds::CsPad::ColumnsPerASIC };
  enum { MaxRowsPerASIC = Pds::CsPad::MaxRowsPerASIC };
  uint16_t gainMap[ColumnsPerASIC][MaxRowsPerASIC];

  CsPadGainMapCfg_Data& operator=(const Pds::CsPad::CsPadGainMapCfg& o);
};

//
// Helper type for Pds::CsPad::ConfigV1QuadReg
//
struct CsPadConfigV1QuadReg_Data  {
  uint32_t                  shiftSelect;
  uint32_t                  edgeSelect;
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
  CsPadReadOnlyCfg_Data     readOnly;
  CsPadDigitalPotsCfg_Data  digitalPots;
  CsPadGainMapCfg_Data      gainMap;

  CsPadConfigV1QuadReg_Data& operator=(const Pds::CsPad::ConfigV1QuadReg& o);
};

//
// Helper type for Pds::CsPad::ConfigV1
//
struct CsPadConfigV1_Data  {
  enum { MaxQuadsPerSensor = Pds::CsPad::MaxQuadsPerSensor };
  uint32_t          runDelay;
  uint32_t          eventCode;
  uint32_t          activeRunMode;
  uint32_t          testDataIndex;
  uint32_t          payloadPerQuad;
  uint32_t          badAsicMask;
  uint32_t          asicMask;
  uint32_t          quadMask;
  CsPadConfigV1QuadReg_Data quads[MaxQuadsPerSensor];
};

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

  CsPadConfigV1_Data m_data ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_CSPADCONFIGV1_H
