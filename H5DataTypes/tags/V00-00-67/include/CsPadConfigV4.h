#ifndef H5DATATYPES_CSPADCONFIGV4_H
#define H5DATATYPES_CSPADCONFIGV4_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadConfigV4.
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
#include "H5DataTypes/CsPadConfigV3.h"
#include "hdf5pp/Group.h"
#include "hdf5pp/Type.h"
#include "pdsdata/cspad/ConfigV4.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {
 
//
// Helper type for Pds::CsPad::ConfigV1QuadReg
//
struct CsPadConfigV2QuadReg_Data  {
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
  uint32_t                  ampReset;
  uint32_t                  digCount;
  uint32_t                  digPeriod;
  CsPadReadOnlyCfg_Data     readOnly;
  CsPadDigitalPotsCfg_Data  digitalPots;
  CsPadGainMapCfg_Data      gainMap;

  CsPadConfigV2QuadReg_Data& operator=(const Pds::CsPad::ConfigV2QuadReg& o);
};

//
// Helper type for Pds::CsPad::ConfigV4
//

class CsPadConfigV4  {
public:

  typedef Pds::CsPad::ConfigV4 XtcType ;

  CsPadConfigV4 () {}
  CsPadConfigV4 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

private:

  enum { MaxQuadsPerSensor = Pds::CsPad::MaxQuadsPerSensor };
  enum { SectionsPerQuad = Pds::CsPad::ASICsPerQuad/2 };
  enum { SectionsTotal = MaxQuadsPerSensor*SectionsPerQuad };
  uint32_t          concentratorVersion;
  uint32_t          runDelay;
  uint32_t          eventCode;
  CsPadProtectionSystemThreshold_Data protectionThresholds[MaxQuadsPerSensor];
  uint32_t          protectionEnable;
  uint32_t          inactiveRunMode;
  uint32_t          activeRunMode;
  uint32_t          testDataIndex;
  uint32_t          payloadPerQuad;
  uint32_t          badAsicMask0;
  uint32_t          badAsicMask1;
  uint32_t          asicMask;
  uint32_t          quadMask;
  uint8_t           roiMask[MaxQuadsPerSensor];
  CsPadConfigV2QuadReg_Data quads[MaxQuadsPerSensor];
  int8_t sections[MaxQuadsPerSensor][SectionsPerQuad];

};

} // namespace H5DataTypes

#endif // H5DATATYPES_CSPADCONFIGV4_H
