#ifndef H5DATATYPES_ENCODERDATAV2_H
#define H5DATATYPES_ENCODERDATAV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EncoderDataV2.
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
#include "pdsdata/encoder/DataV2.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

struct EncoderDataV2_Data {
  uint32_t _33mhz_timestamp;
  uint32_t encoder_count[3];
};

class EncoderDataV2  {
public:

  typedef Pds::Encoder::DataV2 XtcType ;

  // Default constructor
  EncoderDataV2 () {}
  EncoderDataV2 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof xtc ; }

protected:

private:

  EncoderDataV2_Data m_data ;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_ENCODERDATAV2_H
