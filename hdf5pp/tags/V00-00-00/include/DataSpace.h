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

  // Destructor
  ~DataSpace () ;

  /// get number of elements in dataspace
  unsigned size() const ;

  /// Get data space ID
  hid_t dsId() const { return *m_id ; }

protected:

  // constructor
  DataSpace ( hid_t dsid ) ;

private:

  // deleter for  boost smart pointer
  struct DataSpacePtrDeleter {
    void operator()( hid_t* id ) {
      if ( id ) H5Sclose ( *id );
      delete id ;
    }
  };

  // Data members
  boost::shared_ptr<hid_t> m_id ;

};

} // namespace hdf5pp

#endif // HDF5PP_DATASPACE_H
