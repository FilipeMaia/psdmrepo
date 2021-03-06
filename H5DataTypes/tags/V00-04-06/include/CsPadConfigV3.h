#ifndef H5DATATYPES_CSPADCONFIGV3_H
#define H5DATATYPES_CSPADCONFIGV3_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadConfigV3.
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
#include "H5DataTypes/CsPadConfigV1.h"
#include "hdf5pp/Group.h"
#include "hdf5pp/Type.h"
#include "pdsdata/cspad/ConfigV3.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {
  
//
// Helper type for Pds::CsPad::ProtectionSystemThreshold
//
struct CsPadProtectionSystemThreshold  {
  CsPadProtectionSystemThreshold() {}
  CsPadProtectionSystemThreshold(const Pds::CsPad::ProtectionSystemThreshold& o);

  static hdf5pp::Type native_type() ;

private:
  uint32_t adcThreshold;
  uint32_t pixelCountThreshold;
};

//
// Helper type for Pds::CsPad::ConfigV3
//
class CsPadConfigV3  {
public:

  typedef Pds::CsPad::ConfigV3 XtcType ;

  CsPadConfigV3 () {}
  CsPadConfigV3 ( const XtcType& data ) ;

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
  CsPadProtectionSystemThreshold protectionThresholds[MaxQuadsPerSensor];
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
  CsPadConfigV1QuadReg quads[MaxQuadsPerSensor];
  int8_t sections[MaxQuadsPerSensor][SectionsPerQuad];
};

} // namespace H5DataTypes

#endif // H5DATATYPES_CSPADCONFIGV3_H
