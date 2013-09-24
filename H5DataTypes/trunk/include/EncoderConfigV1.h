#ifndef H5DATATYPES_ENCODERCONFIGV1_H
#define H5DATATYPES_ENCODERCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EncoderConfigV1.
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
#include "pdsdata/psddl/encoder.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Encoder::ConfigV1
//
class EncoderConfigV1  {
public:

  typedef Pds::Encoder::ConfigV1 XtcType ;

  EncoderConfigV1 () {}
  EncoderConfigV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

private:

  uint32_t chan_num;
  uint32_t count_mode;
  uint32_t quadrature_mode;
  uint32_t input_num;
  uint32_t input_rising;
  uint32_t ticks_per_sec;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_ENCODERCONFIGV1_H
