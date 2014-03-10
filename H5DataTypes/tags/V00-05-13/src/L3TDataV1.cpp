//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class L3TDataV1...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/L3TDataV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/TypeTraits.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

//----------------
// Constructors --
//----------------
L3TDataV1::L3TDataV1 (const XtcType& data)
  : accept(data.accept())
{
}

hdf5pp::Type
L3TDataV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
L3TDataV1::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<L3TDataV1>() ;
  type.insert_native<uint32_t>( "accept", offsetof(L3TDataV1,accept) ) ;
  return type ;
}

} // namespace H5DataTypes
