#ifndef H5DATATYPES_FLIFRAMEV1_H
#define H5DATATYPES_FLIFRAMEV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FliFrameV1.
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
#include "pdsdata/fli/FrameV1.hh"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Ipimb::DataV1
//
class FliFrameV1  {
public:

  typedef Pds::Fli::FrameV1 XtcType ;

  FliFrameV1 () {}
  FliFrameV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static hdf5pp::Type stored_data_type(uint32_t height, uint32_t width) ;

private:

  uint32_t shotIdStart;
  float    readoutTime;
  float    temperature;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_FLIFRAMEV1_H
