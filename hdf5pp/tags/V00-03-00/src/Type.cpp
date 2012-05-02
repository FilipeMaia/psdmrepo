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
#include "hdf5pp/Exceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // deleter for  boost smart pointer
  struct TypePtrDeleter {
    TypePtrDeleter( bool doClose ) : m_doClose(doClose) {}
    void operator()( hid_t* id ) {
      if ( id and m_doClose ) H5Tclose ( *id );
      delete id ;
    }
    bool m_doClose ;
  };

}

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

/// return size of the type in bytes
size_t
Type::size() const
{
  size_t size = H5Tget_size( *m_id ) ;
  if ( size == 0 ) {
    throw Hdf5CallException( ERR_LOC, "H5Tget_size" ) ;
  }
  return size ;
}

/// get type class, one of the H5T_class_t enum values like H5T_INTEGER,
/// H5T_ARRAY, etc.
H5T_class_t
Type::tclass() const
{
  return H5Tget_class(*m_id);
}

/// get type from which this type was derived
Type
Type::super() const
{
  hid_t tid = H5Tget_super(*m_id) ;
  if ( tid < 0 ) {
    throw Hdf5CallException( ERR_LOC, "H5Tget_super" ) ;
  }
  return UnlockedType( tid ) ;
}

/// make unlocked copy of the type
Type
Type::copy() const
{
  hid_t tid = H5Tcopy(*m_id) ;
  if ( tid < 0 ) {
    throw Hdf5CallException( ERR_LOC, "H5Tcopy" ) ;
  }
  return UnlockedType( tid ) ;
}

/// set type size
void
Type::set_size( size_t size )
{
  if ( H5Tset_size( *m_id, size ) ) {
    throw Hdf5CallException( ERR_LOC, "H5Tset_size" ) ;
  }
}

/// set type precision
void
Type::set_precision( size_t precision )
{
  if ( H5Tset_precision( *m_id, precision ) ) {
    throw Hdf5CallException( ERR_LOC, "H5Tset_precision" ) ;
  }
}

} // namespace hdf5pp
