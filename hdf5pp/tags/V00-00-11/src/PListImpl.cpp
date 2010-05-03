//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PListImpl...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "hdf5pp/PListImpl.h"

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

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace hdf5pp {


//----------------
// Constructors --
//----------------
PListImpl::PListImpl ( hid_t cls )
  : m_id()
{
  m_id = H5Pcreate ( cls ) ;
  if ( m_id < 0 ) throw Hdf5CallException ( "PListImpl", "H5Pcreate" ) ;
}

// copy constructor
PListImpl::PListImpl ( const PListImpl& o )
  : m_id()
{
  m_id = H5Pcopy ( o.m_id ) ;
  if ( m_id < 0 ) throw Hdf5CallException ( "PListImpl", "H5Pcopy" ) ;
}

//--------------
// Destructor --
//--------------
PListImpl::~PListImpl ()
{
  H5Pclose(m_id) ;
}

// assignment
PListImpl&
PListImpl::operator = ( const PListImpl& o )
{
  if ( &o != this ) {
    if ( H5Pclose(m_id) < 0 ) throw Hdf5CallException ( "PListImpl", "H5Pclose" ) ;
    m_id = H5Pcopy ( o.m_id ) ;
    if ( m_id < 0 ) throw Hdf5CallException ( "PListImpl", "H5Pcopy" ) ;
  }
  return *this ;
}


} // namespace hdf5pp
