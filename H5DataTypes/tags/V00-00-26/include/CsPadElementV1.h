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
#include "hdf5pp/Group.h"
#include "pdsdata/cspad/ElementV1.hh"
#include "pdsdata/cspad/ConfigV1.hh"

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
struct CsPadElementHeader_Data  {
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

class CsPadElementV1  {
public:

  typedef Pds::CsPad::ElementV1 XtcType ;
  typedef Pds::CsPad::ConfigV1 ConfigXtcType ;

  CsPadElementV1 () {}
  CsPadElementV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type(unsigned nQuad) ;
  static hdf5pp::Type native_type(unsigned nQuad) ;

  static hdf5pp::Type stored_data_type(unsigned nQuad, unsigned nSect) ;

private:

  CsPadElementHeader_Data m_data ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_CSPADELEMENTV1_H
