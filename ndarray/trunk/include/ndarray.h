#ifndef NDARRAY_NDARRAY_H
#define NDARRAY_NDARRAY_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ndarray.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <functional>
#include <numeric>
#include <iterator>

//----------------------
// Base Class Headers --
//----------------------
#include "nd_elem_access.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/// @addtogroup ndarray

/**
 *  @ingroup ndarray
 */
namespace ndns {

  /**
   *   Enum defines assumed order of the element in memory. Values are used
   *   as parameters for constructors and reshape() method. By default methods
   *   assume C-style memory ordering where last index changes faster. If
   *   in-memory data has Fortran layout (first index changes faster) than
   *   one needs to provide Fortran as the last argument to the above methods.
   *   It is also possible to change strides to any other memory layout
   *   by using strides() method.
   */
  enum Order { C, Fortran };

}

/// @addtogroup ndarray_details

/**
 *  @ingroup ndarray_details
 */
namespace ndarray_details {

  // special destroyer for non-owned data
  template <typename ElemType>
  struct _no_delete {
    void operator()(ElemType*) const {}
  };

  // special destroyer for owned array data
  template <typename ElemType>
  struct _array_delete {
    void operator()(ElemType* ptr) const { delete [] ptr; }
  };

  // some template magic to drop const from type decl
  template <typename T>
  struct unconst { typedef T type; };
  template <typename T>
  struct unconst<const T> { typedef T type; };

}

/// @addtogroup ndarray

/**
 *  @ingroup ndarray
 *
 *  @brief N-dimensional array class.
 *
 *  Ndarray (short for N-dimensional array) class provides high-level C++
 *  interface for multi-dimensional array data. This is a class template with
 *  two parameters - element type and array rank. Array dimensionality
 *  (rank) is fixed at compile time and cannot change. Actual dimensions and
 *  size of array are dynamic and can change (for example in assignment).
 *
 *  The type of the array elements is determined by the first template argument
 *  and it can be any regular C++ type of fixed size. There is a distinction
 *  between const and non-const template arguments. If non-const type is
 *  used for template argument (such as ndarray<int,2>) then the resulting
 *  ndarray is a modifiable object, one could use its methods to get access to
 *  array elements and modify them (through non-const references or pointers).
 *  On the other hand if template argument is a const type (e.g. ndarray<const int,2>)
 *  then array is non-modifiable, it only returns const pointers and references to
 *  the data which cannot be used to modify the data.
 *
 *  There are two essential characteristics of every array - array shape and strides.
 *  Array shape defines the size of every dimension of the array; shape is itself
 *  a 1-dimensional array of size @c NDim (ndarray rank). Shape of the array is set in
 *  the constructor and can be queried with @c shape() method. Strides define memory
 *  layout of the array data. By default (if you do not provide strides in constructor)
 *  ndarray assumes C-type memory layout (last index changes fastest) and
 *  calculates appropriate strides itself. One can provide non-standard strides
 *  for different (even disjoint) memory layouts. ndarray uses strides to calculate
 *  element offset w.r.t. first element of the array. For example for 3-dimensional
 *  array it finds an element as:
 *
 *    @code
 *    ndarr[i][j][k] = *(data + i*stride[0] + j*stride[1] * k*stride[2])
 *    @endcode
 *
 *  where @c data is a pointer to first array element. For C-type memory layout
 *  strides array is calculated from shape array as:
 *
 *    @code
 *    strides[NDim-1] = 1;
 *    for (i = NDim-1 .. 1) strides[i-1] = strides[i]*shape[i];
 *    @endcode
 *
 *  One can query the strides array with the @c strides() method which returns a pointer
 *  to the 1-dimensional array of @c NDim size.
 *
 *  Ndarray can be used to either provide access to the already existing data
 *  (which are not in the form of the ndarray), or to create new objects
 *  with newly allocated memory for array data. Ndarray transparently supports
 *  memory management of the data area, in most cases one should not care about
 *  how memory is allocated or deallocated. There are few different ways to
 *  construct ndarray instances which determine how ndarray manages its corresponding
 *  data:
 *
 *  1. from raw pointers:
 *     @code
 *     int* ptr = new int[100*100];
 *     unsigned int shape[2] = {100, 100};
 *     ndarray<int,2> array(ptr, shape);
 *     //or
 *     ndarray<int,2> array = make_ndarray(ptr, 100, 100);
 *     @endcode
 *
 *     In this case ndarray does not do anything special with the pointer that is
 *     passed to it, it is responsibility of the client code to make sure that
 *     memory is correctly deallocated if necessary and deallocation happens only after
 *     the last copy of ndarray is destroyed. Because of the potential problems it is
 *     not recommended anymore to make ndarrays from raw pointers.
 *
 *  2. internal memory allocation:
 *     @code
 *     unsigned int shape[2] = {100, 100};
 *     ndarray<int,2> array(shape);
 *     // or
 *     ndarray<int,2> array = make_ndarray<int>(100, 100);
 *     @endcode
 *
 *     In this case constructor allocates necessary space to hold array data. This
 *     space is automatically deallocated when the last copy of the ndarray pointing
 *     to that array data is destroyed. This is a preferred way to make ndarrays if
 *     you need to allocate new memory for your data.
 *
 *  3. from shared pointer (for advanced users):
 *     @code
 *     boost::shared_ptr<int> data = ...;
 *     unsigned int shape[2] = {100, 100}
 *     ndarray<int,2> array(data, shape);
 *     // or
 *     ndarray<int,2> array = make_ndarray(data, 100, 100);
 *     @endcode
 *
 *     In this case shared pointer defines memory management policy for the data, ndarray
 *     copies this pointers and shares array data through this pointer with all other
 *     clients. The memory will be deallocated (if necessary) when last copy of shared
 *     pointer disappears (including all copies in ndarray instances). Note that
 *     creating shared pointer for array data needs special care. One cannot just say:
 *     @code
 *     // DO NOT DO THIS, BAD THINGS WILL HAPPEN!
 *     boost::shared_ptr<int> data(new int[100*100]);
 *     @endcode
 *     as this will cause incorrect <tt>operator new</tt> be called when data is going to be
 *     deallocated. Special deleter object (or different way of constructing) is needed
 *     in this case, check boost.shared_ptr documentation.
 *
 *  The main method of accessing array elements is a traditional C-like square
 *  bracket syntax:
 *
 *    @code
 *    ndarray<int, 3> ndarr(...);
 *    int elem = ndarr[i][j][k];
 *    @endcode
 *
 *  Alternatively if the array indices are located in an array one can use @c at() method:
 *
 *    @code
 *    ndarray<int, 3> ndarr(...);
 *    unsigned idx[3] = {i, j, k};
 *    int elem = ndarr.at(idx);
 *    @endcode
 *
 *  One can modify elements of the array if array template type is non-const:
 *
 *    @code
 *    ndarray<int, 3> ndarr(...);
 *    ndarr[i][j][k] = 1000;
 *    @endcode
 *
 *  Additionally ndarray class provides STL-compatible iterators and usual methods
 *  @c begin(), @c end(), @c rbegin(), @c rend(). The iterators can be used with
 *  many standard algorithm. Note that iterators always scan for elements in the
 *  memory order from first (data()) to last (data()+size()) array element
 *  (iterators do not use strides and do not work with disjoint array memory).
 *
 *  Method @c size() returns the total number of elements in array.
 *
 *  Regular ndarray constructor take shape array (and optionally strides array).
 *  It is not always convenient to pass array dimensions as array elements. Few
 *  additional functions are provided which construct ndarray from dimensions
 *  provided via regular arguments, here are few examples of their use:
 *
 *    @code
 *    // Create 1-dim array of size 1048576
 *    int* data = ...;
 *    ndarray<int, 1> arr2d = make_ndarray(data, 1048576);
 *
 *    // Create 2-dim array of dimensions 1024x1024, allocate space for it
 *    ndarray<int, 2> arr2d = make_ndarray<int>(1024, 1024);
 *
 *    // Create 3-dim array of dimensions 4x512x512 from shared pointer
 *    boost::shared_ptr<int> shptr = ...;
 *    ndarray<int, 3> arr2d = make_ndarray(shptr, 4, 512, 512);
 *
 *    // Create non-modifiable array from constant data
 *    const int* cdata = ...;
 *    ndarray<const int, 1> arr2d = make_ndarray(cdata, 1048576);
 *    @endcode
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

template <typename ElemType, unsigned NDim>
class ndarray : public ndarray_details::nd_elem_access<ElemType, NDim> {
  typedef ndarray_details::nd_elem_access<ElemType, NDim> Super;
public:

  typedef ElemType element;
  typedef element* iterator;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef size_t size_type;
  typedef unsigned shape_t;
  typedef int stride_t;
  typedef ndarray<typename ndarray_details::unconst<ElemType>::type, NDim> nonconst_ndarray;

  /// Default constructor makes an empty array
  ndarray () {}

  /**
   *  @brief Constructor that takes pointer to data and shape.
   *
   *  Optional third argument defines memory order of the elements, default is to
   *  assume C order (last index changes fastest). This argument determines how strides
   *  are calculated, strides can be changed later with strides() method.
   *
   *  @param[in] data   Pointer to data array
   *  @param[in] shape  Pointer to dimensions array, size of array is NDim, array data will be copied.
   *  @param[in] order  Memory order of the elements.
   */
  ndarray (ElemType* data, const shape_t* shape, ndns::Order order = ndns::C)
  {
    std::copy(shape, shape+NDim, Super::m_shape);
    _setStrides(order);
    Super::m_data = boost::shared_ptr<ElemType>(data, ndarray_details::_no_delete<ElemType>());
  }

  /**
   *  @brief Constructor that takes shared pointer to data and shape array.
   *
   *  Optional third argument defines memory order of the elements, default is to
   *  assume C order (last index changes fastest). This argument determines how strides
   *  are calculated, strides can be changed later with strides() method.
   *
   *  @param[in] data   Pointer to data array
   *  @param[in] shape  Pointer to dimensions array, size of array is NDim, array data will be copied.
   *  @param[in] order  Memory order of the elements.
   */
  ndarray (const boost::shared_ptr<ElemType>& data, const shape_t* shape, ndns::Order order = ndns::C)
  {
    std::copy(shape, shape+NDim, Super::m_shape);
    _setStrides(order);
    Super::m_data = data;
  }

  /**
   *  @brief Constructor that takes shape array and allocates necessary space for data.
   *
   *  After allocation the data in array is not initialized and will contain garbage.
   *  Optional second argument defines memory order of the elements, default is to
   *  assume C order (last index changes fastest). This argument determines how strides
   *  are calculated, strides can be changed later with strides() method.
   *
   *  @param[in] shape  Pointer to dimensions array, size of array is NDim, array data will be copied.
   *  @param[in] order  Memory order of the elements.
   */
  ndarray (const shape_t* shape, ndns::Order order = ndns::C)
  {
    std::copy(shape, shape+NDim, Super::m_shape);
    _setStrides(order);
    Super::m_data = boost::shared_ptr<ElemType>(new ElemType[size()], ndarray_details::_array_delete<ElemType>());
  }

  /**
   *  @brief Constructor from nd_elem_access_pxy instance.
   *
   *  This constructor is used for slicing of the original ndarray, proxy
   *  is returned from operator[] and you can make ndarray which is a slice
   *  of the original array from that proxy object. Both original array and
   *  this new instance will share the data.
   */
  ndarray (const ndarray_details::nd_elem_access_pxy<ElemType, NDim>& pxy)
  {
    Super::m_data = pxy.m_data;
    std::copy(pxy.m_shape, pxy.m_shape+NDim, Super::m_shape);
    std::copy(pxy.m_strides, pxy.m_strides+NDim, Super::m_strides);
  }

  /**
   *  @brief Copy constructor from other ndarray.
   *
   *  If the template argument (ElemType) is a const type (like <tt>const int</tt> then
   *  this constructor makes const ndarray from non-const, so for example one can write
   *  code which converts non-const array to const:
   *
   *  @code
   *  ndarray<int,3> arr1 = ...;
   *  ndarray<const int,3> arr2(arr1);
   *  @endcode
   *
   *  If template argument (ElemType) is non-const type then this is a regular copy
   *  constructor. It does not actually copy array data, in both cases data is shared
   *  between original array and new instance.
   */
  ndarray(const nonconst_ndarray& other)
    : Super(other)
  {
  }

  /// Assignment operator, data is never copied.
  ndarray& operator=(const nonconst_ndarray& other)
  {
    Super::operator=(other);
    return *this;
  }

  /// Assignment from nd_elem_access_pxy.
  ndarray& operator=(const ndarray_details::nd_elem_access_pxy<ElemType, NDim>& pxy)
  {
    Super::m_data = pxy.m_data;
    std::copy(pxy.m_shape, pxy.m_shape+NDim, Super::m_shape);
    std::copy(pxy.m_strides, pxy.m_strides+NDim, Super::m_strides);
    return *this;
  }


  /**
   *  @brief Array element access.
   *
   *  This method accepts the array of indices, size of the array is the number of
   *  dimensions. Alternative way to access elements in the array is to use regular
   *  operator[] inherited from nd_elem_access base class.
   */
  element& at(shape_t index[]) const {
    element* data = Super::m_data.get();
    for (unsigned i = 0; i != NDim; ++ i) {
      data += index[i]*Super::m_strides[i];
    }
    return *data;
  }

  /**
   *  Returns pointer to the beginning of the data array.
   */
  element* data() const { return Super::m_data.get(); }

  /**
   *  @brief Returns shape of the array.
   *
   *  Returns an array (or pointer to its first element) of the ndarray dimensions.
   */
  const shape_t* shape() const { return Super::m_shape; }

  /**
   *  @brief Returns array strides.
   *
   *  Returns an array (or pointer to its first element) of the ndarray strides.
   */
  const stride_t* strides() const { return Super::m_strides; }

  /**
   *  @brief Changes strides.
   *
   *  If array has a non-conventional memory layout it is possible to change
   *  the strides.
   *
   *  @param[in] strides  Pointer to new strides array, size of array is NDim, array data will be copied.
   */
  void strides(const stride_t* strides) {
    std::copy(strides, strides+NDim, Super::m_strides);
  }

  /**
   *  @brief Changes the shape of the array.
   *
   *  No checks are done on the size of the new array.
   *  Optional second argument defines memory order of the elements, default is to
   *  assume C order (last index changes fastest). This argument determines how strides
   *  are calculated, strides can be changed later with strides() method.
   *
   *  @param[in] shape  Pointer to dimensions array, size of array is NDim, array data will be copied.
   *  @param[in] order  Memory order of the elements.
   */
  void reshape(const shape_t* shape, ndns::Order order = ndns::C) {
    std::copy(shape, shape+NDim, Super::m_shape);
    _setStrides(order);
  }


  /// Returns total number of elements in array
  size_type size() const {
    return std::accumulate(Super::m_shape, Super::m_shape+NDim, size_type(1), std::multiplies<size_type>());
  }

  /// Returns true if array has no data
  bool empty() const { return not Super::m_data; }

  /// Returns iterator to the beginning of data, iteration is performed in memory order
  iterator begin() const { return Super::m_data.get(); }

  /// Returns iterator to the end of data
  iterator end() const { return Super::m_data.get() + size(); }

  /// Returns reverse iterators
  reverse_iterator rbegin() const { return reverse_iterator(end()); }
  reverse_iterator rend() const { return reverse_iterator(begin()); }

  /// swap contents of two arrays
  void swap(ndarray& other) {
    std::swap(Super::m_data, other.Super::m_data);
    std::swap_ranges(Super::m_shape, Super::m_shape+NDim, other.Super::m_shape);
    std::swap_ranges(Super::m_strides, Super::m_strides+NDim, other.Super::m_strides);
  }

  /**
   *  Make a deep copy of the array data, returns modifiable array.
   *
   *  Note that this does not work with disjoint arrays, use on your own risk.
   */
  nonconst_ndarray copy() const {
    nonconst_ndarray res(shape());
    res.strides(strides());
    std::copy(begin(), end(), res.begin());
    return res;
  }

protected:

  // calculate strides from shape array given the assumed ordering
  void _setStrides(ndns::Order order) {
    if (order == ndns::C) {
      Super::m_strides[NDim-1] = 1;
      for (int i = int(NDim)-2; i >= 0; -- i) Super::m_strides[i] = Super::m_strides[i+1] * Super::m_shape[i+1];
    } else {
      Super::m_strides[0] = 1;
      for (unsigned i = 1; i < NDim; ++ i) Super::m_strides[i] = Super::m_strides[i-1] * Super::m_shape[i-1];
    }
  }

private:

};

/// @ingroup ndarray
/// Helper method to create an instance of 1-dimensional array from raw data pointer.
/// Note there may be potential problems with memory management with raw pointers.
template <typename ElemType>
inline
ndarray<ElemType, 1> 
make_ndarray(ElemType* data, unsigned dim0)
{
  unsigned shape[] = {dim0};
  return ndarray<ElemType, 1>(data, shape);
}

/// @ingroup ndarray
/// Helper method to create an instance of 2-dimensional array from raw data pointer.
/// Note there may be potential problems with memory management with raw pointers.
template <typename ElemType>
inline
ndarray<ElemType, 2> 
make_ndarray(ElemType* data, unsigned dim0, unsigned dim1)
{
  unsigned shape[] = {dim0, dim1};
  return ndarray<ElemType, 2>(data, shape);
}

/// @ingroup ndarray
/// Helper method to create an instance of 3-dimensional array from raw data pointer.
/// Note there may be potential problems with memory management with raw pointers.
template <typename ElemType>
inline
ndarray<ElemType, 3> 
make_ndarray(ElemType* data, unsigned dim0, unsigned dim1, unsigned dim2)
{
  unsigned shape[] = {dim0, dim1, dim2};
  return ndarray<ElemType, 3>(data, shape);
}

/// @ingroup ndarray
/// Helper method to create an instance of 4-dimensional array from raw data pointer.
/// Note there may be potential problems with memory management with raw pointers.
template <typename ElemType>
inline
ndarray<ElemType, 4> 
make_ndarray(ElemType* data, unsigned dim0, unsigned dim1, unsigned dim2, unsigned dim3)
{
  unsigned shape[] = {dim0, dim1, dim2, dim3};
  return ndarray<ElemType, 4>(data, shape);
}

/// @ingroup ndarray
/// Helper method to create an instance of 1-dimensional array.
/// Memory for data is allocated internally in this case.
template <typename ElemType>
inline
ndarray<ElemType, 1>
make_ndarray(unsigned dim0)
{
  unsigned shape[] = {dim0};
  return ndarray<ElemType, 1>(shape);
}

/// @ingroup ndarray
/// Helper method to create an instance of 2-dimensional array.
/// Memory for data is allocated internally in this case.
template <typename ElemType>
inline
ndarray<ElemType, 2>
make_ndarray(unsigned dim0, unsigned dim1)
{
  unsigned shape[] = {dim0, dim1};
  return ndarray<ElemType, 2>(shape);
}

/// @ingroup ndarray
/// Helper method to create an instance of 3-dimensional array.
/// Memory for data is allocated internally in this case.
template <typename ElemType>
inline
ndarray<ElemType, 3>
make_ndarray(unsigned dim0, unsigned dim1, unsigned dim2)
{
  unsigned shape[] = {dim0, dim1, dim2};
  return ndarray<ElemType, 3>(shape);
}

/// @ingroup ndarray
/// Helper method to create an instance of 4-dimensional array.
/// Memory for data is allocated internally in this case.
template <typename ElemType>
inline
ndarray<ElemType, 4>
make_ndarray(unsigned dim0, unsigned dim1, unsigned dim2, unsigned dim3)
{
  unsigned shape[] = {dim0, dim1, dim2, dim3};
  return ndarray<ElemType, 4>(shape);
}

/// @ingroup ndarray
/// Helper method to create an instance of 1-dimensional array from shared pointer.
/// Shaed pointer defines memory management policy in this case.
template <typename ElemType>
inline
ndarray<ElemType, 1>
make_ndarray(const boost::shared_ptr<ElemType>& data, unsigned dim0)
{
  unsigned shape[] = {dim0};
  return ndarray<ElemType, 1>(data, shape);
}

/// @ingroup ndarray
/// Helper method to create an instance of 2-dimensional array from shared pointer.
/// Shaed pointer defines memory management policy in this case.
template <typename ElemType>
inline
ndarray<ElemType, 2>
make_ndarray(const boost::shared_ptr<ElemType>& data, unsigned dim0, unsigned dim1)
{
  unsigned shape[] = {dim0, dim1};
  return ndarray<ElemType, 2>(data, shape);
}

/// @ingroup ndarray
/// Helper method to create an instance of 3-dimensional array from shared pointer.
/// Shaed pointer defines memory management policy in this case.
template <typename ElemType>
inline
ndarray<ElemType, 3>
make_ndarray(const boost::shared_ptr<ElemType>& data, unsigned dim0, unsigned dim1, unsigned dim2)
{
  unsigned shape[] = {dim0, dim1, dim2};
  return ndarray<ElemType, 3>(data, shape);
}

/// @ingroup ndarray
/// Helper method to create an instance of 4-dimensional array from shared pointer.
/// Shaed pointer defines memory management policy in this case.
template <typename ElemType>
inline
ndarray<ElemType, 4>
make_ndarray(const boost::shared_ptr<ElemType>& data, unsigned dim0, unsigned dim1, unsigned dim2, unsigned dim3)
{
  unsigned shape[] = {dim0, dim1, dim2, dim3};
  return ndarray<ElemType, 4>(data, shape);
}

#include "nd_format.h"

#endif // NDARRAY_NDARRAY_H
