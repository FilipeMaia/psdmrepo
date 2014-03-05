#ifndef H5DATATYPES_CSPAD2X2ELEMENTV1_H
#define H5DATATYPES_CSPAD2X2ELEMENTV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2ElementV1.
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
#include "hdf5pp/Type.h"
#include "pdsdata/psddl/cspad2x2.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::CsPad2x2::ElementV1
//
class CsPad2x2ElementV1  {
public:

  typedef Pds::CsPad2x2::ElementV1 XtcType ;

  CsPad2x2ElementV1 () {}
  CsPad2x2ElementV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static hdf5pp::Type stored_data_type() ;
  static hdf5pp::Type cmode_data_type() ;

private:

  enum { SbTempSize = 4 };

  uint32_t tid;
  uint32_t seq_count;
  uint32_t ticks;
  uint32_t fiducials;
  uint16_t acq_count;
  uint16_t sb_temp[SbTempSize];
  uint8_t virtual_channel;
  uint8_t lane;
  uint8_t op_code;
  uint8_t quad;
  uint8_t frame_type;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_CSPAD2X2ELEMENTV1_H
