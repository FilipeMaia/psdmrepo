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
#include "H5DataTypes/CsPadElementHeader.h"
#include "hdf5pp/Type.h"
#include "pdsdata/cspad/ElementV1.hh"

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

  CsPadElementHeader m_data ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_CSPADELEMENTV1_H
