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
#include "Lusi/Lusi.h"

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

// make a an array type of any rank
ArrayType
ArrayType::arrayType( const Type& baseType, unsigned rank, hsize_t dims[] )
{
  hid_t tid = H5Tarray_create2( baseType.id(), rank, dims ) ;
  if ( tid < 0 ) throw Hdf5CallException ( "ArrayType::arrayType", "H5Tarray_create2" ) ;
  return ArrayType ( tid ) ;
}

} // namespace hdf5pp
