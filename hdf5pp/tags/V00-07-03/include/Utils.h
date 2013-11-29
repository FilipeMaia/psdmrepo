#ifndef HDF5PP_UTILS_H
#define HDF5PP_UTILS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Utils.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/ArrayType.h"
#include "hdf5pp/DataSet.h"
#include "hdf5pp/Group.h"
#include "hdf5pp/VlenType.h"
#include "ndarray/ndarray.h"

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
 *  @brief Set of utility methods to facilitate work with HDF5 library
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class Utils  {
public:

  // Special deleter for shared pointer, does not call destructor
  template <typename T>
  struct Unmalloc {
    void operator()(T* p) const { free(p); }
  };

  /**
   *  @brief Read one object from dataset
   *
   *  Reads one object from scalar or 1-dimensional dataset. If dataset is scalar then
   *  index must be set to -1, any other value means that dataset is 1-dimensional
   *  and an object at given index will be read. Template parameter defines the type
   *  of the returned object, this class must define static method native_type()
   *  which returns hdf5pp::Type object.
   *
   *  @param[in] ds    dataset object
   *  @param[in] index Object index, if negative then dataset must be scalar
   *
   *  @throw hdf5pp::Exception
   */
  template <typename Data>
  static boost::shared_ptr<Data> readDataSet(hdf5pp::DataSet ds, hsize_t index = -1)
  {
    hdf5pp::DataSpace file_dsp = ds.dataSpace();
    if (index != hsize_t(-1)) file_dsp.select_single(index);
    boost::shared_ptr<Data> ptr = boost::make_shared<Data>();
    ds.read(hdf5pp::DataSpace::makeScalar(), file_dsp, ptr.get(), TypeTraits<Data>::native_type());
    return ptr;
  }


  /**
   *  @brief Read ndarray from dataset.
   *
   *  Reads one object from 1-dimensional dataset or the whole dataset.
   *  if index is not specified then whole dataset is read, ndarray rank
   *  must be the same as dataset rank. If index is specified (and is not -1)
   *  the dataset must have rank 1, single element of dataset will be read
   *  and dataset type must be array with the same rank as ndarray rank
   *  or VLEN array (then Rank must be 1).
   *
   *  @param[in] ds    dataset object
   *  @param[in] index Object index, if negative then whole dataset is read.
   *
   *  @throw hdf5pp::Exception
   */
  template <typename Data, unsigned Rank>
  static ndarray<Data, Rank> readNdarray(hdf5pp::DataSet ds, hsize_t index = -1)
  {
    hdf5pp::DataSpace file_dsp = ds.dataSpace();
    hdf5pp::DataSpace mem_dsp;
    Type memType;

    hsize_t dims[Rank];
    if (index == hsize_t(-1)) {

      // read whole dataset, has to know its rank and dimensions
      
      if (file_dsp.get_simple_extent_type() == H5S_NULL) {
      
        // translator saves empty datasets with H5S_NULL dataspace
        // in this case create 0-sized array
        std::fill_n(dims, Rank, hsize_t(0));
        
      } else {
       
        unsigned rank = file_dsp.rank();
  
        // check rank
        if (rank != Rank) throw Hdf5RankMismatch(ERR_LOC, Rank, rank);
  
        file_dsp.dimensions(dims);
        mem_dsp = DataSpace::makeSimple(Rank, dims, dims);
  
        memType = TypeTraits<Data>::native_type();
        
      }

    } else {

      // select single item for dataset
      file_dsp.select_single(index);
      mem_dsp = DataSpace::makeScalar();

      // read one element from rank-1 dataset, element of a dataset must
      // have array type or VLEN type

      hdf5pp::Type etype = ds.type();
      if (etype.tclass() == H5T_VLEN) {

        // read whole VLEN, Rank must be 1
        if (Rank != 1) throw Hdf5RankMismatch(ERR_LOC, Rank, 1);

        // Memory type is VLEN
        memType = VlenType::vlenType(TypeTraits<Data>::native_type());

        // read data in VLEN structure
        hvl_t vl_data;
        ds.read(mem_dsp, file_dsp, &vl_data, memType);

        // steal a pointer, it has to be free()d
        boost::shared_ptr<Data> shptr(static_cast<Data*>(vl_data.p), Unmalloc<Data>());

        unsigned shape[] = { vl_data.len };
        return ndarray<Data, Rank>(shptr, shape);

      }

      // this will throw if type of data is not an array
      ArrayType type = ArrayType(ds.type());

      // check array type rank, get dimensions
      unsigned rank = type.rank();
      if (rank != Rank) throw Hdf5RankMismatch(ERR_LOC, Rank, rank);
      type.dimensions(dims);

      memType = ArrayType::arrayType(TypeTraits<Data>::native_type(), Rank, dims);
    }

    // make ndarray
    unsigned shape[Rank];
    std::copy(dims, dims+Rank, shape);
    ndarray<Data, Rank> array(shape);

    if (array.size() > 0) {
      // read it
      ds.read(mem_dsp, file_dsp, array.data(), memType);
    }
    
    return array;

  }


  /**
   *  @brief Read object from a named dataset.
   *
   *  Meaning and operation are the same as for readDataSet() method, the difference
   *  is that dataset with the given name and parent is open first, then readDataSet()
   *  is called on that object.
   *
   *  @param[in] group   Group object, parent of the dataset.
   *  @param[in] dataset Dataset name
   *  @param[in] index   Object index, if negative then dataset must be scalar
   *
   *  @throw hdf5pp::Exception
   */
  template <typename Data>
  static boost::shared_ptr<Data> readGroup(hdf5pp::Group group, const std::string& dataset, hsize_t index = -1)
  {
    return readDataSet<Data>(group.openDataSet(dataset), index);
  }

  /**
   *  @brief Read ndarray from a named dataset.
   *
   *  Meaning and operation are the same as for readNdarray(Dataset) method, the difference
   *  is that dataset with the given name and parent is open first, then readDataSet()
   *  is called on that object.
   *
   *  @param[in] group   Group object, parent of the dataset.
   *  @param[in] dataset Dataset name
   *  @param[in] index   Object index, if negative then dataset must be scalar
   *
   *  @throw hdf5pp::Exception
   */
  template <typename Data, unsigned Rank>
  static ndarray<Data, Rank> readNdarray(hdf5pp::Group group, const std::string& dataset, hsize_t index = -1)
  {
    return readNdarray<Data, Rank>(group.openDataSet(dataset), index);
  }

  /**
   *  @brief Store an object in a dataset in a group.
   *
   *  Creates new scalar dataset with a given name and stores data in it. Dataset must not exist
   *  yet, otherwise an exception will be thrown.
   *
   *  @param[in] group   Group object, parent of the dataset.
   *  @param[in] dataset Dataset name
   *  @param[in] data    Object to store
   *  @param[in] native_type    In-memory type of the data
   *  @param[in] stored_type    Type of the data as stored in file
   *
   *  @throw hdf5pp::Exception
   */
  template <typename Data>
  static void storeScalar(hdf5pp::Group group, const std::string& dataset, const Data& data,
      const Type& native_type = TypeTraits<Data>::native_type(),
      const Type& stored_type = TypeTraits<Data>::stored_type())
  {
    _storeScalar(group, dataset, static_cast<const void*>(&data), native_type, stored_type);
  }

  /**
   *  @brief Store an object at specified index in a dataset in a group.
   *
   *  Appends data object to a dataset which has to be rank-1 dataset. The dataset
   *  has to be created first with createDataset() method. If index is negative then
   *  new object is appended to a dataset (dataset size is extended by 1). Otherwise
   *  if index exceeds current dataset size then dataset size is extended to index+1
   *  and all new entries are zero-filled. New object is then stored at specified index.
   *
   *  @param[in] group   Group object, parent of the dataset.
   *  @param[in] dataset Dataset name
   *  @param[in] data    Object to store
   *  @param[in] index   Position in the dataset.
   *  @param[in] native_type    In-memory type of the data
   *
   *  @throw hdf5pp::Exception
   */
  template <typename Data>
  static void storeAt(hdf5pp::Group group, const std::string& dataset, const Data& data,
      long index, const Type& native_type = TypeTraits<Data>::native_type())
  {
    _storeAt(group, dataset, static_cast<const void*>(&data), index, native_type);
  }

  /**
   *  @brief Store ndarray in a dataset in a group.
   *
   *  Creates new aray dataset with a given name and stores data in it. Dataset must not exist
   *  yet, otherwise an exception will be thrown.
   *
   *  @param[in] group   Group object, parent of the dataset.
   *  @param[in] dataset Dataset name
   *  @param[in] data    Object to store
   *  @param[in] native_type    In-memory type of the data
   *  @param[in] stored_type    Type of the data as stored in file
   *
   *  @throw hdf5pp::Exception
   */
  template <typename ElemType, unsigned NDim>
  static void storeNDArray(hdf5pp::Group group, const std::string& dataset, const ndarray<ElemType, NDim>& array,
      const Type& native_type = TypeTraits<ElemType>::native_type(),
      const Type& stored_type = TypeTraits<ElemType>::stored_type())
  {
    _storeArray(group, dataset, static_cast<const void*>(array.data()), NDim, array.shape(), native_type, stored_type);
  }

  /**
   *  @brief Store ndarray at specified index in a dataset in a group.
   *
   *  Appends data object to a dataset which has to be rank-1 dataset. The dataset
   *  has to be created first with createDataset() method. If index is negative then
   *  new object is appended to a dataset (dataset size is extended by 1). Otherwise
   *  if index exceeds current dataset size then dataset size is extended to index+1
   *  and all new entries are zero-filled. New object is then stored at specified index.
   *
   *  @param[in] group   Group object, parent of the dataset.
   *  @param[in] dataset Dataset name
   *  @param[in] data    Object to store
   *  @param[in] index   Position in the dataset.
   *  @param[in] native_type    In-memory type of the array elements
   *
   *  @throw hdf5pp::Exception
   */
  template <typename ElemType, unsigned NDim>
  static void storeNDArrayAt(hdf5pp::Group group, const std::string& dataset, const ndarray<ElemType, NDim>& array,
      long index, const Type& native_type = TypeTraits<ElemType>::native_type())
  {
    std::vector<hsize_t> dims(array.shape(), array.shape()+NDim);
    ArrayType array_native = ArrayType::arrayType(native_type, NDim, &dims.front());
    _storeAt(group, dataset, static_cast<const void*>(array.data()), index, array_native);
  }

  /**
   *  @brief Create a chunked rank=1 dataset.
   *
   *  @param[in] group         Group object, parent of the dataset.
   *  @param[in] dataset       Dataset name
   *  @param[in] stored_type   Type of the data as stored in dataset
   *  @param[in] chunk_size    Size of single chunk in objects of type stored_type.
   *  @param[in] chunk_cache_size  Size of chunk cache in number of chunks
   *  @param[in] deflate       Compression level, negative disables compression
   *  @param[in] shuffle       If set to true then shuffle filter is enabled.
   *  @return    Dataset instance
   *  @throw hdf5pp::Exception
   */
  static DataSet createDataset(hdf5pp::Group group, const std::string& dataset, const Type& stored_type,
      hsize_t chunk_size, hsize_t chunk_cache_size, int deflate, bool shuffle);

  /**
   *  @brief Resize rank-1 dataset.
   *
   *  Extends dataset to a new size (has to be a rank-1 dataset). If size given as argument
   *  is negative this is equivalent to extending the size by one element. New added elements
   *  will be default-initialized (zero-filled unless dataset type defines special default values).
   *  This method can also be used to shrink dataset size.
   *
   *  @param[in] group         Group object, parent of the dataset.
   *  @param[in] dataset       Dataset name
   *  @param[in] size          New dataset size, negative means append one element.
   *  @throw hdf5pp::Exception
   */
  static void resizeDataset(hdf5pp::Group group, const std::string& dataset, long size);

private:

  /// template-free implementation of storeAt()
  static void _storeAt(hdf5pp::Group group, const std::string& dataset, const void* data, long index,
      const Type& native_type);

  /// template-free implementation of storeScalar()
  static void _storeScalar(hdf5pp::Group group, const std::string& dataset, const void* data,
      const Type& native_type, const Type& stored_type);

  /// template-free implementation of storeArray()
  static void _storeArray(hdf5pp::Group group, const std::string& dataset, const void* data,
      unsigned rank, const unsigned* shape, const Type& native_type, const Type& stored_type);

};

} // namespace hdf5pp

#endif // HDF5PP_UTILS_H
