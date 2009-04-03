#ifndef HDF5PP_DATASET_H
#define HDF5PP_DATASET_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DataSet.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------


//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Attribute.h"
#include "hdf5pp/DataSpace.h"
#include "hdf5pp/Exceptions.h"
#include "hdf5pp/TypeTraits.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  C++ class for HDF5 data set.
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace hdf5pp {

template <typename T>
class DataSet  {
public:

  // Destructor
  ~DataSet () ;

  /// create attribute for this dataset
  template <typename U>
  Attribute<U> createAttr ( const std::string& name, const DataSpace& dspc = DataSpace::makeScalar() ) {
    return Attribute<U>::createAttr ( *m_id, name, dspc ) ;
  }

  /// open existing attribute
  template <typename U>
  Attribute<U> openAttr ( const std::string& name ) {
    return Attribute<U>::openAttr ( *m_id, name ) ;
  }


protected:

  // Constructor
  DataSet ( hid_t id, const DataSpace& dspc ) : m_dspc(dspc), m_id( new hid_t(id), DataSetPtrDeleter() ) {}

  /// create new dataset
  DataSet createDataSet ( hid_t parent, const std::string& name, const DataSpace& dspc ) {
    hid_t ds = H5Dcreate2 ( parent, name.c_str(), TypeTraits<T>::h5type_native(), dspc.dsId(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ) ;
    if ( ds < 0 ) throw Hdf5CallException( "DataSet::createDataSet", "H5Dcreate2" ) ;
    return DataSet ( ds, dspc ) ;
  }

  /// open existing dataset
  DataSet openDataSet ( hid_t parent, const std::string& name ) {
    hid_t ds = H5Dopen2 ( parent, name.c_str(), H5P_DEFAULT ) ;
    if ( ds < 0 ) throw Hdf5CallException( "DataSet::openDataSet", "H5Dopen2" ) ;
    hid_t dspc = H5Dget_space( ds ) ;
    if ( dspc < 0 ) throw Hdf5CallException( "DataSet::openDataSet", "H5Dget_space" ) ;
    return DataSpace ( ds, dspc ) ;
  }


private:

  // deleter for  boost smart pointer
  struct DataSetPtrDeleter {
    void operator()( hid_t* id ) {
      if ( id ) H5Aclose ( *id );
      delete id ;
    }
  };

  // Data members
  DataSpace m_dspc;
  boost::shared_ptr<hid_t> m_id ;

};

} // namespace hdf5pp

#endif // HDF5PP_DATASET_H
