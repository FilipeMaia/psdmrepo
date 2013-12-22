#ifndef H5DATATYPES_PRINCETONFRAMEV2_H
#define H5DATATYPES_PRINCETONFRAMEV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PrincetonFrameV2.
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
// Helper type for Pds::Ipimb::DataV2
//
class PrincetonFrameV2  {
public:

  typedef Pds::Princeton::FrameV2 XtcType ;

  PrincetonFrameV2 () {}
  PrincetonFrameV2 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static hdf5pp::Type stored_data_type(uint32_t height, uint32_t width) ;

private:

  uint32_t shotIdStart;
  float    readoutTime;
  float    temperature;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_PRINCETONFRAMEV2_H
