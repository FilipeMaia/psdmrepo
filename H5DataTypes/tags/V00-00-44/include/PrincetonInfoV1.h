#ifndef H5DATATYPES_PRINCETONINFOV1_H
#define H5DATATYPES_PRINCETONINFOV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PrincetonInfoV1.
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

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "hdf5pp/Group.h"
#include "pdsdata/princeton/InfoV1.hh"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Princeton::InfoV1
//
struct PrincetonInfoV1_Data  {
  float temperature;
};

class PrincetonInfoV1  {
public:

  typedef Pds::Princeton::InfoV1 XtcType ;

  PrincetonInfoV1 () {}
  PrincetonInfoV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

private:

  PrincetonInfoV1_Data m_data ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_PRINCETONINFOV1_H
