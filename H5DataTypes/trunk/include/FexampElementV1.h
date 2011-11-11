#ifndef H5DATATYPES_FEXAMPELEMENTV1_H
#define H5DATATYPES_FEXAMPELEMENTV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FexampElementV1.
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
#include "pdsdata/fexamp/ElementV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper class for Pds::Fexamp::ElementV1
//
class FexampElementV1  {
public:

  typedef Pds::Fexamp::ElementV1 XtcType ;
  typedef Pds::Fexamp::ConfigV1 ConfigXtcType ;

  FexampElementV1 () {}
  FexampElementV1(const XtcType& data, const ConfigXtcType& config) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static hdf5pp::Type stored_data_type() ;
  static hdf5pp::Type cmode_data_type() ;

private:

  uint32_t m_seq_count;
  uint32_t m_tid;
  uint16_t m_acq_count;
  uint8_t m_virtual_channel;
  uint8_t m_lane;
  uint8_t m_op_code;
  uint8_t m_elementId;
  uint8_t m_frame_type;
  uint32_t m_penultimateWord;
  uint32_t m_ultimateWord;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_FEXAMPELEMENTV1_H
