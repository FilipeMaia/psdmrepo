#ifndef H5DATATYPES_PRINCETONCONFIGV4_H
#define H5DATATYPES_PRINCETONCONFIGV4_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PrincetonConfigV4.
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
#include "pdsdata/psddl/princeton.ddl.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Princeton::ConfigV4
//
class PrincetonConfigV4  {
public:

  typedef Pds::Princeton::ConfigV4 XtcType ;

  PrincetonConfigV4 () {}
  PrincetonConfigV4 ( const XtcType& data ) ;

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
  uint32_t maskedHeight;
  uint32_t kineticHeight;
  float    vsSpeed;
  float    exposureTime;
  float    coolingTemp;
  uint8_t  gainIndex;
  uint8_t  readoutSpeedIndex;
  uint16_t exposureEventCode;
  uint32_t numDelayShots;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_PRINCETONCONFIGV4_H
