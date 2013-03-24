#ifndef HDF5PP_DATASPACE_H
#define HDF5PP_DATASPACE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DataSpace.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5/hdf5.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace hdf5pp {

/// @addtogroup hdf5pp

/**
 *  @ingroup hdf5pp
 *
 *  Class for data space.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class DataSpace  {
public:

  /// Create new scalar dataspace
  static DataSpace makeScalar () ;

  /// Create new simple dataspace
  static DataSpace makeSimple ( int rank, const hsize_t * dims, const hsize_t * maxdims ) ;

  /// Create new simple rank-1 dataspace
  static DataSpace makeSimple ( hsize_t dim, hsize_t maxdim ) ;

  /// Create new NULL dataspace
  static DataSpace makeNull () ;

  /// Create new H5S_ALL dataspace/selection
  static DataSpace makeAll () ;

  // constructor
  DataSpace () {}

  // constructor
  explicit DataSpace ( hid_t dsid ) ;

  // Destructor
  ~DataSpace () ;

  /// Hyperslab selection, returns this object
  DataSpace select_hyperslab ( H5S_seloper_t op,
                              const hsize_t *start,
                              const hsize_t *stride,
                              const hsize_t *count,
                              const hsize_t *block ) ;

  /// Selection which includes single element from rank-1 dataset, returns this object
  DataSpace select_single(hsize_t index) ;
  
  /// Get the type of the dataspace, returns one of the H5S_SCALAR, H5S_SIMPLE, or H5S_NULL
  H5S_class_t get_simple_extent_type() const; 

  /// get the rank of the data space
  unsigned rank() const ;

  // Get dataspace dimensions, size of dims array must be at least rank().
  // Either dims or maxdims can be zero pointers.
  void dimensions(hsize_t* dims, hsize_t* maxdims = 0);

  /// get number of elements in data space
  unsigned size() const ;

  /// Get data space ID
  hid_t id() const { return *m_id ; }

  // close the data space
  void close() ;

  // returns true if there is a real object behind
  bool valid() const { return m_id.get() ; }

protected:

private:

  // Data members
  boost::shared_ptr<hid_t> m_id ;

};

} // namespace hdf5pp

#endif // HDF5PP_DATASPACE_H
