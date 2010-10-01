#ifndef H5DATATYPES_CSPADConfigV2_H
#define H5DATATYPES_CSPADConfigV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadConfigV2.
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
#include "pdsdata/cspad/ConfigV2.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {
  
//
// Helper type for Pds::CsPad::ConfigV2
//
struct CsPadConfigV2_Data  {
  enum { MaxQuadsPerSensor = Pds::CsPad::MaxQuadsPerSensor };
  enum { SectionsPerQuad = Pds::CsPad::ASICsPerQuad/2 };
  enum { SectionsTotal = MaxQuadsPerSensor*SectionsPerQuad };
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
  uint8_t           roiMask[MaxQuadsPerSensor];
  CsPadConfigV1QuadReg_Data quads[MaxQuadsPerSensor];
  int8_t sections[MaxQuadsPerSensor][SectionsPerQuad];
};

class CsPadConfigV2  {
public:

  typedef Pds::CsPad::ConfigV2 XtcType ;

  CsPadConfigV2 () {}
  CsPadConfigV2 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

private:

  CsPadConfigV2_Data m_data ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_CSPADConfigV2_H
