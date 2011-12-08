//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PListFileAccess...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "hdf5pp/PListFileAccess.h"

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
PListFileAccess::PListFileAccess ()
  : m_impl(H5P_FILE_ACCESS)
{
}

//--------------
// Destructor --
//--------------
PListFileAccess::~PListFileAccess ()
{
}

// use family driver
void
PListFileAccess::set_family_driver ( hsize_t memb_size, const PListFileAccess& memb_fapl )
{
  herr_t stat = H5Pset_fapl_family ( m_impl.id(), memb_size, memb_fapl.plist() ) ;
  if ( stat < 0 ) {
    throw Hdf5CallException ( "PListFileAccess::set_family_driver", "H5Pset_fapl_family" ) ;
  }
}

} // namespace hdf5pp
