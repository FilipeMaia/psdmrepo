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
#include "hdf5pp/CompoundType.h"

//-----------------
// C/C++ Headers --
//-----------------
#include "hdf5/hdf5.h"
#include "hdf5pp/ArrayType.h"
#include "hdf5pp/Exceptions.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace hdf5pp {

CompoundType
CompoundType::compoundType( size_t size )
{
  hid_t tid = H5Tcreate ( H5T_COMPOUND, size ) ;
  if ( tid < 0 ) throw Hdf5CallException ( ERR_LOC, "H5Tcreate" ) ;
  return CompoundType ( tid ) ;
}

// add one more member
void
CompoundType::insert ( const char* name, size_t offset, const Type& t, size_t size )
{
  Type mtype = t;
  if (size > 0) {
    mtype = hdf5pp::ArrayType::arrayType(mtype, size);
  }
  herr_t stat = H5Tinsert ( id(), name, offset, mtype.id() ) ;
  if ( stat < 0 ) throw Hdf5CallException ( ERR_LOC, "H5Tinsert" ) ;
}

} // namespace hdf5pp
