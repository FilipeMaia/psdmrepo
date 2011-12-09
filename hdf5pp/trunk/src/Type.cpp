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
    throw Hdf5CallException( "Type::size", "H5Tget_size" ) ;
  }
  return size ;
}

/// make unlocked copy of the type
Type
Type::copy() const
{
  hid_t tid = H5Tcopy(*m_id) ;
  if ( tid < 0 ) {
    throw Hdf5CallException( "Type::copy", "H5Tcopy" ) ;
  }
  return UnlockedType( tid ) ;
}

/// set type size
void
Type::set_size( size_t size )
{
  if ( H5Tset_size( *m_id, size ) ) {
    throw Hdf5CallException( "Type::set_size", "H5Tset_size" ) ;
  }
}

/// set type precision
void
Type::set_precision( size_t precision )
{
  if ( H5Tset_precision( *m_id, precision ) ) {
    throw Hdf5CallException( "Type::set_precision", "H5Tset_precision" ) ;
  }
}

} // namespace hdf5pp
