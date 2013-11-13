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
#include "pdsdata/psddl/princeton.ddl.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Princeton::InfoV1
//
class PrincetonInfoV1  {
public:

  typedef Pds::Princeton::InfoV1 XtcType ;

  PrincetonInfoV1 () {}
  PrincetonInfoV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

private:

  float temperature;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_PRINCETONINFOV1_H
