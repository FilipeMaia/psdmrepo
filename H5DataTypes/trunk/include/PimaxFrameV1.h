#ifndef H5DATATYPES_PIMAXFRAMEV1_H
#define H5DATATYPES_PIMAXFRAMEV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PimaxFrameV1.
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
#include "pdsdata/psddl/pimax.ddl.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Ipimb::DataV1
//
class PimaxFrameV1  {
public:

  typedef Pds::Pimax::FrameV1 XtcType ;

  PimaxFrameV1 () {}
  PimaxFrameV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static hdf5pp::Type stored_data_type(uint32_t height, uint32_t width) ;

private:

  uint32_t shotIdStart;
  float    readoutTime;
  float    temperature;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_PIMAXFRAMEV1_H
