#ifndef H5DATATYPES_CSPADElementV2_H
#define H5DATATYPES_CSPADElementV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadElementV2.
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
#include "H5DataTypes/CsPadElementV1.h"
#include "hdf5pp/Group.h"
#include "pdsdata/cspad/ElementV2.hh"
#include "pdsdata/cspad/ConfigV2.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

namespace H5DataTypes {
  
class CsPadElementV2  {
public:

  typedef Pds::CsPad::ElementV2 XtcType ;
  typedef Pds::CsPad::ConfigV2 ConfigXtcType ;

  CsPadElementV2 () {}
  CsPadElementV2 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type(unsigned nQuad) ;
  static hdf5pp::Type native_type(unsigned nQuad) ;

  static hdf5pp::Type stored_data_type(unsigned nSect) ;

private:

  CsPadElementHeader_Data m_data ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_CSPADElementV2_H
