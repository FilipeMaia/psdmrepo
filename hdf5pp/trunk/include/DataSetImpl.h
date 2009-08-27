#ifndef HDF5PP_DATASETIMPL_H
#define HDF5PP_DATASETIMPL_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DataSetImpl.
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
#include "hdf5pp/PListDataSetCreate.h"
#include "hdf5pp/Type.h"

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

class DataSetImpl  {
public:


  /// create new data set, specify the type explicitly
  static DataSetImpl createDataSet ( hid_t parent,
                                     const std::string& name,
                                     const Type& type,
                                     const DataSpace& dspc,
                                     const PListDataSetCreate& plistDScreate ) ;

  /// open existing dataset
  static DataSetImpl openDataSet ( hid_t parent, const std::string& name ) ;

  // Default constructor
  DataSetImpl () {}

  // Destructor
  ~DataSetImpl () {}

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

  /// Changes the sizes of a dataset’s dimensions.
  void set_extent ( const hsize_t size[] ) ;

  // store the data
  void store ( const Type& memType,
               const DataSpace& memDspc,
               const DataSpace& fileDspc,
               const void* data ) ;

  // close the data set
  void close() { m_id.reset() ; }

  /// access data space
  DataSpace dataSpace() ;

  // returns true if there is a real object behind
  bool valid() const { return m_id.get() ; }

protected:

  DataSetImpl ( hid_t id ) ;

private:

  // deleter for  boost smart pointer
  struct DataSetPtrDeleter {
    void operator()( hid_t* id ) {
      if ( id ) H5Dclose ( *id );
      delete id ;
    }
  };

  // Data members
  boost::shared_ptr<hid_t> m_id ;

};

} // namespace hdf5pp

#endif // HDF5PP_DATASETIMPL_H
