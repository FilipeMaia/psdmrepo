#ifndef H5DATATYPES_CSPADELEMENTV1_H
#define H5DATATYPES_CSPADELEMENTV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadElementV1.
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
#include "pdsdata/psddl/cspad.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

namespace H5DataTypes {
  
//
// Helper type for Pds::CsPad::ElementV1
//
class CsPadElementV1  {
public:

  typedef Pds::CsPad::ElementV1 XtcType ;

  CsPadElementV1 () {}
  CsPadElementV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type(unsigned nQuad) ;
  static hdf5pp::Type native_type(unsigned nQuad) ;

  static hdf5pp::Type stored_data_type(unsigned nQuad, unsigned nSect) ;
  static hdf5pp::Type cmode_data_type(unsigned nQuad, unsigned nSect) ;

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

#endif // H5DATATYPES_CSPADELEMENTV1_H
