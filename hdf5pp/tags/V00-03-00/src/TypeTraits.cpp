//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TypeTraits...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "hdf5pp/TypeTraits.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/ArrayType.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace hdf5pp {

Type
TypeTraitsHelper::string_h5type (size_t size)
{
  static hid_t string_h5type_inst ;
  if (size == 0) {
    // return variable-size string type
    static bool initialized = false ;
    if ( not initialized ) {
      initialized = true ;
      string_h5type_inst = H5Tcopy ( H5T_C_S1 ) ;
      H5Tset_size( string_h5type_inst, H5T_VARIABLE ) ;
      H5Tlock ( string_h5type_inst ) ;
    }
    return Type::LockedType(string_h5type_inst) ;
  } else {
    // return fixed-size string type
    hid_t type = H5Tcopy ( H5T_C_S1 ) ;
    H5Tset_size(type, size);
    return Type::UnlockedType(type);
  }
}

Type
TypeTraitsHelper::sized_h5type(const Type& type, size_t size)
{
  if (size == 0) return type;
  return ArrayType::arrayType(type, size);
}

} // namespace hdf5pp
