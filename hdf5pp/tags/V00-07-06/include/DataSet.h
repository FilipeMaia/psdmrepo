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
#include "hdf5pp/PListDataSetAccess.h"
#include "hdf5pp/PListDataSetCreate.h"
#include "hdf5pp/Type.h"
#include "hdf5pp/TypeTraits.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace hdf5pp {

class Group ;

/// @addtogroup hdf5pp

/**
 *  @ingroup hdf5pp
 *
 *  C++ class for HDF5 data set.
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class DataSet {
public:

  // Default constructor
  DataSet() {}

  // Destructor
  ~DataSet () {}

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

  /// Changes the sizes of a dataset's dimensions.
  void set_extent ( const hsize_t size[] );

  /// same operation for rank-1 data set
  void set_extent ( hsize_t size ) { set_extent(&size); }

  // store the data
  template <typename T>
  void store ( const DataSpace& memDspc,
               const T* data,
               const hdf5pp::Type& native_type = TypeTraits<T>::native_type() )
  {
    _store( native_type, memDspc, DataSpace::makeAll(), TypeTraits<T>::address( *data ) ) ;
  }

  // store the data, give file dataspace
  template <typename T>
  void store ( const DataSpace& memDspc,
               const DataSpace& fileDspc,
               const T* data,
               const hdf5pp::Type& native_type = TypeTraits<T>::native_type() )
  {
    _store( native_type, memDspc, fileDspc, TypeTraits<T>::address( *data ) ) ;
  }

  // store the data, give file dataspace
  void store ( const DataSpace& memDspc,
               const DataSpace& fileDspc,
               const void* data,
               const hdf5pp::Type& native_type)
  {
    _store( native_type, memDspc, fileDspc, data);
  }

  // retrieve the data from dataset
  template <typename T>
  void read (const DataSpace& memDspc,
             const DataSpace& fileDspc,
             T* data,
             const hdf5pp::Type& native_type = TypeTraits<T>::native_type())
  {
    _read(native_type, memDspc, fileDspc, TypeTraits<T>::address(*data));
  }

  // reclaim space allocated to vlen structures
  template <typename T>
  void vlen_reclaim(const DataSpace& memDspc,
                    T* data,
                    const hdf5pp::Type& native_type = TypeTraits<T>::native_type())
  {
    _vlen_reclaim(native_type, memDspc, TypeTraits<T>::address(*data));
  }

  // close the data set
  void close();

  /// access data space
  DataSpace dataSpace();

  /// get chunk size, this method only works for 1-dim datasets
  size_t chunkSize() const;

  /// access dataset type
  Type type();

  // returns true if there is a real object behind
  bool valid() const { return m_id; }

  // get dataset name, this will likely be a relative name
  std::string name() const;

  // get dataset id
  hid_t id() const { return *m_id; }

protected:

  friend class Group ;

  // Constructor
  DataSet(hid_t id);

  /// create new data set, type is determined at run time
  static DataSet createDataSet(hid_t parent,
                               const std::string& name,
                               const Type& type,
                               const DataSpace& dspc,
                               const PListDataSetCreate& plistDScreate,
                               const PListDataSetAccess& plistDSaccess);

  /// open existing dataset
  static DataSet openDataSet(hid_t parent,
      const std::string& name,
      const PListDataSetAccess& plistDSaccess);

private:

  // store the data
  void _store( const Type& memType,
               const DataSpace& memDspc,
               const DataSpace& fileDspc,
               const void* data ) ;

  // read the data
  void _read(const Type& memType,
             const DataSpace& memDspc,
             const DataSpace& fileDspc,
             void* data);

  void _vlen_reclaim(const hdf5pp::Type& type, const DataSpace& memDspc, void* data);

  // Data members
  boost::shared_ptr<hid_t> m_id ;

};

} // namespace hdf5pp

#endif // HDF5PP_DATASET_H
