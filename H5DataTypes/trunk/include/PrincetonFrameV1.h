#ifndef H5DATATYPES_PRINCETONFRAMEV1_H
#define H5DATATYPES_PRINCETONFRAMEV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PrincetonFrameV1.
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
#include "pdsdata/princeton/FrameV1.hh"
#include "pdsdata/princeton/ConfigV1.hh"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Ipimb::DataV1
//
struct PrincetonFrameV1_Data  {
  uint32_t shotIdStart;
  float    readoutTime;
};

class PrincetonFrameV1  {
public:

  typedef Pds::Princeton::FrameV1 XtcType ;
  typedef Pds::Princeton::ConfigV1 ConfigXtcType ;

  PrincetonFrameV1 () {}
  PrincetonFrameV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static hdf5pp::Type stored_data_type( const ConfigXtcType& config ) ;

private:

  PrincetonFrameV1_Data m_data ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_PRINCETONFRAMEV1_H
