#ifndef H5DATATYPES_PRINCETONCONFIGV3_H
#define H5DATATYPES_PRINCETONCONFIGV3_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PrincetonConfigV3.
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
#include "pdsdata/princeton/ConfigV3.hh"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Princeton::ConfigV3
//
class PrincetonConfigV3  {
public:

  typedef Pds::Princeton::ConfigV3 XtcType ;

  PrincetonConfigV3 () {}
  PrincetonConfigV3 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

private:

  uint32_t width;
  uint32_t height;
  uint32_t orgX;
  uint32_t orgY;
  uint32_t binX;
  uint32_t binY;
  float    exposureTime;
  float    coolingTemp;
  uint8_t  gainIndex;
  uint8_t  readoutSpeedIndex;
  uint16_t exposureEventCode;
  uint32_t numDelayShots;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_PRINCETONCONFIGV3_H
