#ifndef H5DATATYPES_CSPADELEMENTHEADER_H
#define H5DATATYPES_CSPADELEMENTHEADER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadElementHeader.
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
#include "hdf5pp/Type.h"
#include "pdsdata/cspad/ElementHeader.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::CsPad::ElementHeader
//

class CsPadElementHeader  {
public:

  typedef Pds::CsPad::ElementHeader XtcType ;

  CsPadElementHeader () {}
  CsPadElementHeader ( const XtcType& data ) ;

  // Destructor
  ~CsPadElementHeader () ;

  static hdf5pp::Type native_type() ;

protected:

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

#endif // H5DATATYPES_CSPADELEMENTHEADER_H
