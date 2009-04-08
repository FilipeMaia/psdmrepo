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

/// Create new simple rank-1 dataspace
DataSpace
DataSpace::makeSimple ( hsize_t dim, hsize_t maxdim )
{
  return makeSimple ( 1, &dim, &maxdim ) ;
}

/// Create new NULL dataspace
DataSpace
DataSpace::makeNull ()
{
  hid_t dsid = H5Screate ( H5S_NULL ) ;
  if ( dsid < 0 ) throw Hdf5CallException( "DataSpace::makeSimple", "H5Screate") ;
  return DataSpace( dsid ) ;
}

/// Create new H5S_ALL dataspace/selection
DataSpace
DataSpace::makeAll ()
{
  return DataSpace( H5S_ALL ) ;
}

/// Hyperslab selection
void
DataSpace::select_hyperslab ( H5S_seloper_t op,
                              const hsize_t *start,
                              const hsize_t *stride,
                              const hsize_t *count,
                              const hsize_t *block )
{
  herr_t stat = H5Sselect_hyperslab ( *m_id, op, start, stride, count, block ) ;
  if ( stat < 0 ) throw Hdf5CallException( "DataSpace::select_hyperslab", "H5Sselect_hyperslab") ;
}


/// get the rank of the data space
unsigned
DataSpace::rank() const
{
  int rank = H5Sget_simple_extent_ndims(*m_id) ;
  if ( rank < 0 ) throw Hdf5CallException( "DataSpace::rank", "H5Sget_simple_extent_ndims") ;
  return rank ;
}

/// get number of elements in dataspace
unsigned
DataSpace::size() const
{
  hssize_t size = H5Sget_simple_extent_npoints(*m_id) ;
  return size ;
}

// close the data space
void
DataSpace::close()
{
  m_id.reset();
}

} // namespace hdf5pp
