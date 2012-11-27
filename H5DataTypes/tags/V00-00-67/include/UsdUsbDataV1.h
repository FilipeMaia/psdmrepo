#ifndef H5DATATYPES_USDUSBDATAV1_H
#define H5DATATYPES_USDUSBDATAV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class UsdUsbDataV1.
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
#include "pdsdata/usdusb/DataV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

class UsdUsbDataV1  {
public:

  typedef Pds::UsdUsb::DataV1 XtcType ;

  // Default constructor
  UsdUsbDataV1 () {}
  UsdUsbDataV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof xtc ; }

protected:

private:

  enum { Encoder_Inputs = Pds::UsdUsb::DataV1::Encoder_Inputs };
  enum { Analog_Inputs  = Pds::UsdUsb::DataV1::Analog_Inputs };

  uint32_t encoder_count[Encoder_Inputs];
  uint16_t analog_in[Analog_Inputs];
  uint32_t timestamp;
  uint8_t  digital_in;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_USDUSBDATAV1_H
