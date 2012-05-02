//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: Type.cpp 250 2009-04-08 01:02:05Z salnikov $
//
// Description:
//	Class Type...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "hdf5pp/ArrayType.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5/hdf5.h"
#include "hdf5pp/Exceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace hdf5pp {

// cast base type to array type, will throw exception if base type
// is not an array.
ArrayType::ArrayType(const Type& baseType)
  : Type(baseType)
{
  if (baseType.tclass() != H5T_ARRAY) {
    throw Hdf5BadTypeCast(ERR_LOC, "ArrayType");
  }
}

// make a an array type of any rank
ArrayType
ArrayType::arrayType( const Type& baseType, unsigned rank, const hsize_t dims[] )
{
  hid_t tid = H5Tarray_create2( baseType.id(), rank, dims ) ;
  if ( tid < 0 ) throw Hdf5CallException ( ERR_LOC, "H5Tarray_create2" ) ;
  return ArrayType ( tid ) ;
}

// get array rank
int
ArrayType::rank() const
{
  return H5Tget_array_ndims(id());
}

// get array dimensions, size of dims array must be at least rank()
void
ArrayType::dimensions(hsize_t dims[])
{
  int err = H5Tget_array_dims2(id(), dims);
  if (err < 0) throw Hdf5CallException ( ERR_LOC, "H5Tget_array_dims2" ) ;
}

} // namespace hdf5pp
