#ifndef H5DATATYPES_ENCODERCONFIGV2_H
#define H5DATATYPES_ENCODERCONFIGV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EncoderConfigV2.
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
#include "pdsdata/encoder/ConfigV2.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Encoder::ConfigV2
//
class EncoderConfigV2  {
public:

  typedef Pds::Encoder::ConfigV2 XtcType ;

  EncoderConfigV2 () {}
  EncoderConfigV2 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

private:

  uint32_t chan_mask;
  uint32_t count_mode;
  uint32_t quadrature_mode;
  uint32_t input_num;
  uint32_t input_rising;
  uint32_t ticks_per_sec;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_ENCODERCONFIGV2_H
