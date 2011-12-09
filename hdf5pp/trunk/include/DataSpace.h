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

/**
 *  Class for data space.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace hdf5pp {

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

  /// Hyperslab selection
  void select_hyperslab ( H5S_seloper_t op,
                          const hsize_t *start,
                          const hsize_t *stride,
                          const hsize_t *count,
                          const hsize_t *block ) ;

  /// get the rank of the data space
  unsigned rank() const ;

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
