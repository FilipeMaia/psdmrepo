#ifndef H5DATATYPES_CSPAD2X2CONFIGV2_H
#define H5DATATYPES_CSPAD2X2CONFIGV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2ConfigV2.
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
#include "pdsdata/psddl/cspad2x2.ddl.h"
#include "H5DataTypes/CsPad2x2ConfigV1.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {
  
//
// Helper type for Pds::CsPad2x2::ConfigV2QuadReg
//
struct CsPad2x2ConfigV2QuadReg  {
  CsPad2x2ConfigV2QuadReg() {}
  CsPad2x2ConfigV2QuadReg(const Pds::CsPad2x2::ConfigV2QuadReg& o);

  static hdf5pp::Type native_type() ;

private:
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
  uint32_t                  biasTuning;
  uint32_t                  pdpmndnmBalance;
  CsPad2x2ReadOnlyCfg       readOnly;
  CsPad2x2DigitalPotsCfg    digitalPots;
  CsPad2x2GainMapCfg        gainMap;
};

//
// Helper type for Pds::CsPad2x2::ConfigV2
//
class CsPad2x2ConfigV2  {
public:

  typedef Pds::CsPad2x2::ConfigV2 XtcType ;

  CsPad2x2ConfigV2 () {}
  CsPad2x2ConfigV2 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

private:

  CsPad2x2ConfigV2QuadReg quad;
  uint32_t          testDataIndex;
  CsPad2x2ProtectionSystemThreshold protectionThreshold;
  uint32_t          protectionEnable;
  uint32_t          inactiveRunMode;
  uint32_t          activeRunMode;
  uint32_t          runTriggerDelay;
  uint32_t          payloadSize;
  uint32_t          badAsicMask;
  uint32_t          asicMask;
  uint32_t          roiMask;
  uint32_t          numAsicsRead;
  uint32_t          numAsicsStored;
  uint32_t          concentratorVersion;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_CSPAD2X2CONFIGV2_H
