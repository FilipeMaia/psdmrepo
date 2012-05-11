#ifndef H5DATATYPES_CSPAD2X2CONFIGV1_H
#define H5DATATYPES_CSPAD2X2CONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2ConfigV1.
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
#include "hdf5pp/Type.h"
#include "pdsdata/cspad2x2/ConfigV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {
  
//
// Helper type for Pds::CsPad2x2::CsPad2x2DigitalPotsCfg
//
struct CsPad2x2DigitalPotsCfg_Data  {
  enum { PotsPerQuad = Pds::CsPad2x2::PotsPerQuad };
  uint8_t         pots[PotsPerQuad];

  CsPad2x2DigitalPotsCfg_Data& operator=(const Pds::CsPad2x2::CsPad2x2DigitalPotsCfg& o);
};

//
// Helper type for Pds::CsPad2x2::CsPad2x2ReadOnlyCfg
//
struct CsPad2x2ReadOnlyCfg_Data  {
  uint32_t        shiftTest;
  uint32_t        version;

  CsPad2x2ReadOnlyCfg_Data& operator=(const Pds::CsPad2x2::CsPad2x2ReadOnlyCfg& o);
};

//
// Helper type for Pds::CsPad2x2::CsPad2x2GainMapCfg
//
struct CsPad2x2GainMapCfg_Data  {
  enum { ColumnsPerASIC = Pds::CsPad2x2::ColumnsPerASIC };
  enum { MaxRowsPerASIC = Pds::CsPad2x2::MaxRowsPerASIC };
  uint16_t gainMap[ColumnsPerASIC][MaxRowsPerASIC];

  CsPad2x2GainMapCfg_Data& operator=(const Pds::CsPad2x2::CsPad2x2GainMapCfg& o);
};

//
// Helper type for Pds::CsPad2x2::ConfigV1QuadReg
//
struct CsPad2x2ConfigV1QuadReg_Data  {
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
  uint32_t                  ampReset;
  uint32_t                  digCount;
  uint32_t                  digPeriod;
  uint32_t                  PeltierEnable;
  uint32_t                  kpConstant;
  uint32_t                  kiConstant;
  uint32_t                  kdConstant;
  uint32_t                  humidThold;
  uint32_t                  setPoint;
  CsPad2x2ReadOnlyCfg_Data  readOnly;
  CsPad2x2DigitalPotsCfg_Data digitalPots;
  CsPad2x2GainMapCfg_Data   gainMap;

  CsPad2x2ConfigV1QuadReg_Data& operator=(const Pds::CsPad2x2::ConfigV1QuadReg& o);
};

//
// Helper type for Pds::CsPad2x2::ProtectionSystemThreshold
//
struct CsPad2x2ProtectionSystemThreshold_Data  {
  uint32_t adcThreshold;
  uint32_t pixelCountThreshold;

  CsPad2x2ProtectionSystemThreshold_Data& operator=(const Pds::CsPad2x2::ProtectionSystemThreshold& o);
};

//
// Helper type for Pds::CsPad2x2::ConfigV1
//
struct CsPad2x2ConfigV1_Data  {
  CsPad2x2ConfigV1QuadReg_Data quad;
  uint32_t          testDataIndex;
  CsPad2x2ProtectionSystemThreshold_Data protectionThreshold;
  uint32_t          protectionEnable;
  uint32_t          inactiveRunMode;
  uint32_t          activeRunMode;
  uint32_t          payloadSize;
  uint32_t          badAsicMask;
  uint32_t          asicMask;
  uint32_t          roiMask;
  uint32_t          numAsicsRead;
  uint32_t          numAsicsStored;
  uint32_t          concentratorVersion;
};

class CsPad2x2ConfigV1  {
public:

  typedef Pds::CsPad2x2::ConfigV1 XtcType ;

  CsPad2x2ConfigV1 () {}
  CsPad2x2ConfigV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

private:

  CsPad2x2ConfigV1_Data m_data ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_CSPAD2X2CONFIGV1_H
