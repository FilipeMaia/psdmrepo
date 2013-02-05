#ifndef H5DATATYPES_FCCDCONFIGV1_H
#define H5DATATYPES_FCCDCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FccdConfigV1.
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
#include "pdsdata/fccd/FccdConfigV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::FCCD::FccdConfigV1
//
struct FccdConfigV1_Data  {
  uint32_t width;
  uint32_t height;
  uint32_t trimmedWidth;
  uint32_t trimmedHeight;
  uint16_t outputMode;
};

class FccdConfigV1  {
public:

  typedef Pds::FCCD::FccdConfigV1 XtcType ;

  FccdConfigV1 () {}
  FccdConfigV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

private:

  FccdConfigV1_Data m_data ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_FCCDCONFIGV1_H
