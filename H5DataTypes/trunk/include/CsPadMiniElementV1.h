#ifndef H5DATATYPES_CSPADMINIELEMENTV1_H
#define H5DATATYPES_CSPADMINIELEMENTV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadMiniElementV1.
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
#include "H5DataTypes/CsPadElementHeader.h"
#include "hdf5pp/Type.h"
#include "pdsdata/cspad/MiniElementV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::CsPad::MiniElementV1
//
class CsPadMiniElementV1  {
public:

  typedef Pds::CsPad::MiniElementV1 XtcType ;

  CsPadMiniElementV1 () {}
  CsPadMiniElementV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static hdf5pp::Type stored_data_type() ;
  static hdf5pp::Type cmode_data_type() ;

private:

  CsPadElementHeader m_data ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_CSPADMINIELEMENTV1_H
