#ifndef H5DATATYPES_ENCODERDATAV1_H
#define H5DATATYPES_ENCODERDATAV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EncoderDataV1.
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
#include "pdsdata/encoder/DataV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

struct EncoderDataV1_Data {
  uint32_t _33mhz_timestamp;
  uint32_t encoder_count;
};

class EncoderDataV1  {
public:

  typedef Pds::Encoder::DataV1 XtcType ;

  // Default constructor
  EncoderDataV1 () {}
  EncoderDataV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof xtc ; }

protected:

private:

  EncoderDataV1_Data m_data ;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_ENCODERDATAV1_H
