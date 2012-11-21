#ifndef H5DATATYPES_USDUSBCONFIGV1_H
#define H5DATATYPES_USDUSBCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class UsdUsbConfigV1.
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
#include "pdsdata/usdusb/ConfigV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

class UsdUsbConfigV1  {
public:

  typedef Pds::UsdUsb::ConfigV1 XtcType ;

  // Default constructor
  UsdUsbConfigV1 () {}
  UsdUsbConfigV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static void store ( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof xtc ; }

protected:

private:

  enum { NCHANNELS = Pds::UsdUsb::ConfigV1::NCHANNELS };

  uint32_t counting_mode[NCHANNELS];
  uint32_t quadrature_mode[NCHANNELS];

};

} // namespace H5DataTypes

#endif // H5DATATYPES_USDUSBCONFIGV1_H
