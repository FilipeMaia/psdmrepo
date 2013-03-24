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

namespace {

  // deleter for  boost smart pointer
  struct DataSpacePtrDeleter {
    void operator()( hid_t* id ) {
      if ( id and *id != H5S_ALL ) H5Sclose ( *id );
      delete id ;
    }
  };

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace hdf5pp {

//----------------
// Constructors --
//----------------
DataSpace::DataSpace ( hid_t dsid )
  : m_id( new hid_t(dsid), ::DataSpacePtrDeleter() )
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
  if ( dsid < 0 ) throw Hdf5CallException( ERR_LOC, "H5Screate") ;
  return DataSpace( dsid ) ;
}

/// Create new simple dataspace
DataSpace
DataSpace::makeSimple ( int rank, const hsize_t * dims, const hsize_t * maxdims )
{
  hid_t dsid = H5Screate_simple ( rank, dims, maxdims ) ;
  if ( dsid < 0 ) throw Hdf5CallException( ERR_LOC, "H5Screate_simple") ;
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
  if ( dsid < 0 ) throw Hdf5CallException( ERR_LOC, "H5Screate") ;
  return DataSpace( dsid ) ;
}

/// Create new H5S_ALL dataspace/selection
DataSpace
DataSpace::makeAll ()
{
  return DataSpace( H5S_ALL ) ;
}

/// Hyperslab selection
DataSpace
DataSpace::select_hyperslab ( H5S_seloper_t op,
                              const hsize_t *start,
                              const hsize_t *stride,
                              const hsize_t *count,
                              const hsize_t *block )
{
  herr_t stat = H5Sselect_hyperslab ( *m_id, op, start, stride, count, block ) ;
  if ( stat < 0 ) throw Hdf5CallException( ERR_LOC, "H5Sselect_hyperslab") ;
  return *this;
}

/// Selection which includes single element from rank-1 dataset
DataSpace
DataSpace::select_single(hsize_t index)
{
  hsize_t start[] = { index } ;
  hsize_t size[] = { 1 } ;
  return select_hyperslab ( H5S_SELECT_SET, start, 0, size, 0 );
}

/// Get the type of the dataspace, returns one of the H5S_SCALAR, H5S_SIMPLE, or H5S_NULL
H5S_class_t 
DataSpace::get_simple_extent_type() const
{
  H5S_class_t type = H5Sget_simple_extent_type(*m_id) ;
  if ( type == H5S_NO_CLASS ) throw Hdf5CallException( ERR_LOC, "H5Sget_simple_extent_type") ;
  return type ;
}

/// get the rank of the data space
unsigned
DataSpace::rank() const
{
  int rank = H5Sget_simple_extent_ndims(*m_id) ;
  if ( rank < 0 ) throw Hdf5CallException( ERR_LOC, "H5Sget_simple_extent_ndims") ;
  return rank ;
}

// Get dataspace dimensions, size of dims array must be at least rank().
// Either dims or maxdims can be zero pointers.
void
DataSpace::dimensions(hsize_t* dims, hsize_t* maxdims)
{
  int rank = H5Sget_simple_extent_dims(*m_id, dims, maxdims);
  if ( rank < 0 ) throw Hdf5CallException( ERR_LOC, "H5Sget_simple_extent_dims") ;
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
