#ifndef H5DATATYPES_L3TDATAV1_H
#define H5DATATYPES_L3TDATAV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class L3TDataV1.
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
#include "pdsdata/psddl/l3t.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::L3T::DataV1
//
class L3TDataV1  {
public:

  typedef Pds::L3T::DataV1 XtcType ;

  L3TDataV1 () {}
  L3TDataV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

protected:

private:

  uint32_t accept;
  
};

} // namespace H5DataTypes

#endif // H5DATATYPES_L3TDATAV1_H
