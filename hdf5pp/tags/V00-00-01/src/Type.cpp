//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
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
#include "hdf5pp/Type.h"

//-----------------
// C/C++ Headers --
//-----------------

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

//----------------
// Constructors --
//----------------
Type::Type ()
  : m_id ()
{
}

Type::Type ( hid_t id, bool doClose )
  : m_id ( new hid_t(id), TypePtrDeleter(doClose) )
{
}

//--------------
// Destructor --
//--------------
Type::~Type ()
{
}

CompoundType
CompoundType::compoundType( size_t size )
{
  hid_t tid = H5Tcreate ( H5T_COMPOUND, size ) ;
  if ( tid < 0 ) throw Hdf5CallException ( "CompoundType::compoundType", "H5Tcreate" ) ;
  return CompoundType ( tid ) ;
}

// add one more member
void
CompoundType::insert ( const char* name, size_t offset, Type t )
{
  herr_t stat = H5Tinsert ( id(), name, offset, t.id() ) ;
  if ( stat < 0 ) throw Hdf5CallException ( "CompoundType::compoundType", "H5Tcreate" ) ;
}

// make a an array type of any rank
ArrayType
ArrayType::arrayType( Type baseType, unsigned rank, hsize_t dims[] )
{
  hid_t tid = H5Tarray_create2( baseType.id(), rank, dims ) ;
  if ( tid < 0 ) throw Hdf5CallException ( "ArrayType::arrayType", "H5Tarray_create2" ) ;
  return ArrayType ( tid ) ;
}

} // namespace hdf5pp
