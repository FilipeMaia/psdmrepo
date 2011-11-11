#ifndef H5DATATYPES_GSC16AICONFIGV1_H
#define H5DATATYPES_GSC16AICONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Gsc16aiConfigV1.
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
#include "pdsdata/gsc16ai/ConfigV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Gsc16ai::ConfigV1
//
class Gsc16aiConfigV1  {
public:

  typedef Pds::Gsc16ai::ConfigV1 XtcType ;

  Gsc16aiConfigV1 () {}
  Gsc16aiConfigV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

protected:

private:

  uint16_t    _voltageRange;
  uint16_t    _firstChan;
  uint16_t    _lastChan;
  uint16_t    _inputMode;
  uint16_t    _triggerMode;
  uint16_t    _dataFormat;
  uint16_t    _fps;
  int8_t      _autocalibEnable;
  int8_t      _timeTagEnable;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_GSC16AICONFIGV1_H
