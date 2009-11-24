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
#include "hdf5pp/DataSetImpl.h"
#include "hdf5pp/DataSpace.h"
#include "hdf5pp/PListDataSetCreate.h"
#include "hdf5pp/Type.h"
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

class Group ;

template <typename T>
class DataSet {
public:

  // Default constructor
  DataSet() {}

  // Destructor
  ~DataSet () {}

  /// create attribute for this dataset
  template <typename U>
  Attribute<U> createAttr ( const std::string& name, const DataSpace& dspc = DataSpace::makeScalar() ) {
    return m_impl.createAttr<U> ( name, dspc ) ;
  }

  /// open existing attribute
  template <typename U>
  Attribute<U> openAttr ( const std::string& name ) {
    return m_impl.openAttr<U> ( name ) ;
  }

  /// Changes the sizes of a dataset's dimensions.
  void set_extent ( const hsize_t size[] ) {
    m_impl.set_extent ( size ) ;
  }
  /// same operation for rank-1 data set
  void set_extent ( hsize_t size ) {
    m_impl.set_extent ( &size ) ;
  }

  // store the data
  void store ( const DataSpace& memDspc,
               const T* data,
               const hdf5pp::Type& native_type = TypeTraits<T>::native_type() )
  {
    m_impl.store( native_type, memDspc, DataSpace::makeAll(), TypeTraits<T>::address( *data ) ) ;
  }

  // store the data, give file dataspace
  void store ( const DataSpace& memDspc,
               const DataSpace& fileDspc,
               const T* data,
               const hdf5pp::Type& native_type = TypeTraits<T>::native_type() )
  {
    m_impl.store( native_type, memDspc, fileDspc, TypeTraits<T>::address( *data ) ) ;
  }

  // close the data set
  void close() { m_impl.close() ; }

  /// access data space
  DataSpace dataSpace() { return m_impl.dataSpace() ; }

  // returns true if there is a real object behind
  bool valid() const { return m_impl.valid() ; }

protected:

  friend class Group ;

  // Constructor
  DataSet ( DataSetImpl impl ) : m_impl(impl) {}

  /// create new data set, type is determined at run time
  static DataSet createDataSet ( hid_t parent,
                                const std::string& name,
                                const Type& type,
                                const DataSpace& dspc,
                                const PListDataSetCreate& plistDScreate )
  {
    return DataSet ( DataSetImpl::createDataSet ( parent, name, type, dspc, plistDScreate ) ) ;
  }

  /// open existing dataset
  static DataSet openDataSet ( hid_t parent, const std::string& name )
  {
    return DataSet( DataSetImpl::openDataSet ( parent, name ) ) ;
  }

private:

  DataSetImpl m_impl ;

};

} // namespace hdf5pp

#endif // HDF5PP_DATASET_H
