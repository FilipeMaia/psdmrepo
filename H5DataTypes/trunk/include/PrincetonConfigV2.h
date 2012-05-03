#ifndef H5DATATYPES_PRINCETONCONFIGV2_H
#define H5DATATYPES_PRINCETONCONFIGV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PrincetonConfigV2.
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
#include "pdsdata/princeton/ConfigV2.hh"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Princeton::ConfigV2
//
class PrincetonConfigV2  {
public:

  typedef Pds::Princeton::ConfigV2 XtcType ;

  PrincetonConfigV2 () {}
  PrincetonConfigV2 ( const XtcType& data ) ;

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
  uint16_t gainIndex;
  uint16_t readoutSpeedIndex;
  uint16_t readoutEventCode;
  uint16_t delayMode;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_PRINCETONCONFIGV2_H
