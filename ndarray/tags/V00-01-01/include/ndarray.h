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
#include "ndarray/nd_elem_access.h"

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
 *  and it can be any regular C++ type of fixed size.
 *
 *  Ndarray is a wrapper around the existing in-memory data arrays, it cannot
 *  be used to create (allocate) new data arrays, existing pre-allocated
 *  memory pointer must be provided to the constructor of ndarray. Ndarray
 *  provides read-only access to the array data, its primary use is for
 *  accessing data in psana framework which are non-modifiable by design.
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
 *  Additionally ndarray class provides STL-compatible iterators and usual methods
 *  @c begin(), @c end(), @c rbegin(), @c rend(). The iterators can be used with
 *  many standard algorithm. Note that iterators always scan for elements in the
 *  memory order from first (data()) to last (data()+size()) array element
 *  (iterators do not use strides and do not work with disjoint array memory).
 *  Only the const version of iterators are provided as the data in array are read-only.
 *
 *  Method @c size() returns the total number of elements in array.
 *
 *  Regular ndarray constructor take shape array (and optionally strides array).
 *  It is not always convenient to pass array dimensions as array elements. Few
 *  additional functions are provided which construct ndarray from dimensions
 *  provided via regular arguments, here are few examples of their use:
 *
 *    @code
 *    int* data = ...;
 *    // Create 1-dim array of size 1048576
 *    ndarray<int, 1> arr2d = make_ndarray(data, 1048576);
 *    // Create 2-dim array of dimensions 1024x1024
 *    ndarray<int, 2> arr2d = make_ndarray(data, 1024, 1024);
 *    // Create 3-dim array of dimensions 4x512x512
 *    ndarray<int, 3> arr2d = make_ndarray(data, 4, 512, 512);
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
  typedef const ElemType* const_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef size_t size_type;

  /// Default constructor makes an empty array
  ndarray ()
  {
    Super::m_data = 0;
    std::fill_n(Super::m_shape, NDim, 0U);
    std::fill_n(Super::m_strides, NDim, 0U);
  }

  /**
   *  @brief Constructor that takes pointer to data and shape.
   *
   *  @param[in] data   Pointer to data array
   *  @param[in] shape  Pointer to dimensions array, size of array is NDim, array data will be copied.
   */
  ndarray (const ElemType* data, const unsigned* shape)
  {
    Super::m_data = data;
    std::copy(shape, shape+NDim, Super::m_shape);
    // calculate strides for standard C layout
    Super::m_strides[NDim-1] = 1;
    for (int i = NDim-2; i >= 0; -- i) Super::m_strides[i] = Super::m_strides[i+1] * shape[i+1];
  }

  /**
   *  @brief Constructor that takes pointer to data, shape and strides.
   *
   *  @param[in] data   Pointer to data array
   *  @param[in] shape  Pointer to dimensions array, size of array is NDim, array data will be copied.
   *  @param[in] strides Pointer to strides array, size of array is NDim, array data will be copied.
   */
  ndarray (const ElemType* data, const unsigned* shape, const unsigned* strides)
  {
    Super::m_data = data;
    std::copy(shape, shape+NDim, Super::m_shape);
    std::copy(strides, strides+NDim, Super::m_strides);
  }

  /**
   *  @brief Constructor from other ndarray.
   */
  ndarray (const ndarray& other)
  {
    Super::m_data = other.Super::m_data;
    std::copy(other.Super::m_shape, other.Super::m_shape+NDim, Super::m_shape);
    std::copy(other.Super::m_strides, other.Super::m_strides+NDim, Super::m_strides);
  }

  /**
   *  @brief Constructor from other nd_elem_access_pxy.
   */
  ndarray (const ndarray_details::nd_elem_access_pxy<ElemType, NDim>& pxy)
  {
    Super::m_data = pxy.m_data;
    std::copy(pxy.m_shape, pxy.m_shape+NDim, Super::m_shape);
    std::copy(pxy.m_strides, pxy.m_strides+NDim, Super::m_strides);
  }

  // Destructor
  ~ndarray () {}

  /// Assignment operator
  ndarray& operator=(const ndarray& other)
  {
    if (this == &other) return *this;
    Super::m_data = other.Super::m_data;
    std::copy(other.Super::m_shape, other.Super::m_shape+NDim, Super::m_shape);
    std::copy(other.Super::m_strides, other.Super::m_strides+NDim, Super::m_strides);
    return *this;
  }

  /// Assignment from nd_elem_access_pxy
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
  const element& at(unsigned index[]) const {
    const ElemType* data = Super::m_data;
    for (unsigned i = 0; i != NDim; ++ i) {
      data += index[i]*Super::m_strides[i];
    }
    return *data;
  }

  /**
   *  Returns pointer to the beginning of the data array.
   */
  const ElemType* data() const { return Super::m_data; }

  /**
   *  @brief Returns shape of the array.
   *
   *  Returns an array (or pointer to its first element) of the ndarray dimensions.
   */
  const unsigned* shape() const { return Super::m_shape; }

  /**
   *  @brief Returns array strides.
   *
   *  Returns an array (or pointer to its first element) of the ndarray strides.
   */
  const unsigned* strides() const { return Super::m_strides; }

  /// Returns total number of elements in array
  size_type size() const {
    return std::accumulate(Super::m_shape, Super::m_shape+NDim, size_type(1), std::multiplies<size_type>());
  }

  /// Returns true if array has no data
  bool empty() const { return not Super::m_data; }

  /// Returns iterator to the beginning of data, iteration is performed in memory order
  const_iterator begin() const { return Super::m_data; }

  /// Returns iterator to the end of data
  const_iterator end() const { return Super::m_data + size(); }

  /// Returns reverse iterators
  const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
  const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

  /// swap contents of two arrays
  void swap(ndarray& other) {
    std::swap(Super::m_data, other.Super::m_data);
    std::swap_ranges(Super::m_shape, Super::m_shape+NDim, other.Super::m_shape);
    std::swap_ranges(Super::m_strides, Super::m_strides+NDim, other.Super::m_strides);
  }

protected:

private:

};

/// Helper method to create an instance of 1-dimensional array
template <typename ElemType>
inline
ndarray<ElemType, 1> 
make_ndarray(const ElemType* data, unsigned dim0)
{
  unsigned shape[] = {dim0};
  return ndarray<ElemType, 1>(data, shape);
}

/// Helper method to create an instance of 2-dimensional array
template <typename ElemType>
inline
ndarray<ElemType, 2> 
make_ndarray(const ElemType* data, unsigned dim0, unsigned dim1)
{
  unsigned shape[] = {dim0, dim1};
  return ndarray<ElemType, 2>(data, shape);
}

/// Helper method to create an instance of 3-dimensional array
template <typename ElemType>
inline
ndarray<ElemType, 3> 
make_ndarray(const ElemType* data, unsigned dim0, unsigned dim1, unsigned dim2)
{
  unsigned shape[] = {dim0, dim1, dim2};
  return ndarray<ElemType, 3>(data, shape);
}

/// Helper method to create an instance of 4-dimensional array
template <typename ElemType>
inline
ndarray<ElemType, 4> 
make_ndarray(const ElemType* data, unsigned dim0, unsigned dim1, unsigned dim2, unsigned dim3)
{
  unsigned shape[] = {dim0, dim1, dim2, dim3};
  return ndarray<ElemType, 4>(data, shape);
}

#endif // NDARRAY_NDARRAY_H
