#ifndef H5DATATYPES_TIMEPIXDATAV1_H
#define H5DATATYPES_TIMEPIXDATAV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TimepixDataV1.
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
#include "hdf5pp/Type.h"
#include "pdsdata/timepix/DataV1.hh"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Timepix::DataV1
//

class TimepixDataV1  {
public:

  typedef Pds::Timepix::DataV1 XtcType ;

  TimepixDataV1 () {}
  TimepixDataV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static hdf5pp::Type stored_data_type(uint32_t height, uint32_t width) ;

private:

  uint32_t timestamp;
  uint16_t frameCounter;
  uint16_t lostRows;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_TIMEPIXDATAV1_H
