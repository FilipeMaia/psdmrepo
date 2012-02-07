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
#include "H5DataTypes/CsPad2x2ElementHeader.h"
#include "hdf5pp/Type.h"
#include "pdsdata/cspad2x2/ElementV1.hh"

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

  CsPad2x2ElementHeader m_data ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_CSPAD2X2ELEMENTV1_H
