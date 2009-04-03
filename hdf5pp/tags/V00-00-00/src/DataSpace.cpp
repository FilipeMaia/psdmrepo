//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DataSpace...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "hdf5pp/DataSpace.h"

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
DataSpace::DataSpace ( hid_t dsid )
  : m_id( new hid_t(dsid), DataSpacePtrDeleter() )
{
}

//--------------
// Destructor --
//--------------
DataSpace::~DataSpace ()
{
}

/// Create new scalar dataspace
DataSpace DataSpace::makeScalar ()
{
  hid_t dsid = H5Screate ( H5S_SCALAR ) ;
  if ( dsid < 0 ) throw Hdf5CallException( "DataSpace::makeScalar", "H5Screate") ;
  return DataSpace( dsid ) ;
}

/// Create new simple dataspace
DataSpace
DataSpace::makeSimple ( int rank, const hsize_t * dims, const hsize_t * maxdims )
{
  hid_t dsid = H5Screate_simple ( rank, dims, maxdims ) ;
  if ( dsid < 0 ) throw Hdf5CallException( "DataSpace::makeSimple", "H5Screate_simple") ;
  return DataSpace( dsid ) ;
}

/// get number of elements in dataspace
unsigned
DataSpace::size() const
{
  hssize_t size = H5Sget_simple_extent_npoints(*m_id) ;
  return size ;
}


} // namespace hdf5pp
