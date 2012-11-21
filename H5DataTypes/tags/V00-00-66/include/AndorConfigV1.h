#ifndef H5DATATYPES_ANDORCONFIGV1_H
#define H5DATATYPES_ANDORCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AndorConfigV1.
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
#include "pdsdata/andor/ConfigV1.hh"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Andor::ConfigV1
//
class AndorConfigV1  {
public:

  typedef Pds::Andor::ConfigV1 XtcType ;

  AndorConfigV1 () {}
  AndorConfigV1 ( const XtcType& data ) ;

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
  uint8_t  fanMode;
  uint8_t  baselineClamp;
  uint8_t  highCapacity;
  uint8_t  gainIndex;
  uint16_t readoutSpeedIndex;
  uint16_t exposureEventCode;
  uint32_t numDelayShots;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_ANDORCONFIGV1_H
