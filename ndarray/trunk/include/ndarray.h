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
 *  @brief C++ source file code template.
 *
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
  typedef ElemType* iterator;
  typedef const ElemType* const_iterator;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef size_t size_type;

  /// Default constructor makes an empty array
  ndarray ()
  {
    Super::m_data = 0;
    Super::m_own = false;
    std::fill_n(Super::m_shape, NDim, 0U);
    std::fill_n(Super::m_strides, NDim, 0U);
  }

  /**
   *  @brief Constructor that takes pointer to data and shape.
   *
   *  If optional boolean argument 'own' is supplied and is true
   *  then this ndarray object takes ownership of the data pointer,
   *  data will be deleted in the destructor using operator delete [].
   *
   *  @param[in] data   Pointer to data array
   *  @param[in] shape  Pointer to dimensions array, size of array is NDim, array data will be copied.
   *  @param[in] own    If true then data pointer will be deleted in destructor.
   */
  ndarray (ElemType* data, const unsigned* shape, bool own = false)
  {
    Super::m_data = data;
    Super::m_own = own;
    std::copy(shape, shape+NDim, Super::m_shape);
    // calculate strides for standard C layout
    Super::m_strides[NDim-1] = 1;
    for (int i = NDim-2; i >= 0; -- i) Super::m_strides[i] = Super::m_strides[i+1] * shape[i+1];
  }

  /**
   *  @brief Constructor that takes pointer to data, shape and strides.
   *
   *  If optional boolean argument 'own' is supplied and is true
   *  then this ndarray object takes ownership of the data pointer,
   *  data will be deleted in the destructor using operator delete [].
   *
   *  @param[in] data   Pointer to data array
   *  @param[in] shape  Pointer to dimensions array, size of array is NDim, array data will be copied.
   *  @param[in] shape  Pointer to strides array, size of array is NDim, array data will be copied.
   *  @param[in] own    If true then data pointer will be deleted in destructor.
   */
  ndarray (ElemType* data, const unsigned* shape, const unsigned* strides, bool own = false)
  {
    Super::m_data = data;
    Super::m_own = own;
    std::copy(shape, shape+NDim, Super::m_shape);
    std::copy(strides, strides+NDim, Super::m_strides);
  }

  /**
   *  @brief Constructor from other ndarray.
   */
  ndarray (const ndarray& other)
  {
    Super::m_data = other.Super::m_data;
    Super::m_own = false;
    std::copy(other.Super::m_shape, other.Super::m_shape+NDim, Super::m_shape);
    std::copy(other.Super::m_strides, other.Super::m_strides+NDim, Super::m_strides);
  }

  /**
   *  @brief Constructor from other nd_elem_access_pxy.
   */
  ndarray (const ndarray_details::nd_elem_access_pxy<ElemType, NDim>& pxy)
  {
    Super::m_data = pxy.m_data;
    Super::m_own = false;
    std::copy(pxy.m_shape, pxy.m_shape+NDim, Super::m_shape);
    std::copy(pxy.m_strides, pxy.m_strides+NDim, Super::m_strides);
  }

  // Destructor
  ~ndarray () { if (Super::m_own) delete [] Super::m_data; }

  /// Assignment operator
  ndarray& operator=(const ndarray& other)
  {
    if (this == &other) return;
    if (Super::m_own) delete [] Super::m_data;
    Super::m_data = other.Super::m_data;
    Super::m_own = false;
    std::copy(other.Super::m_shape, other.Super::m_shape+NDim, Super::m_shape);
    std::copy(other.Super::m_strides, other.Super::m_strides+NDim, Super::m_strides);
    return *this;
  }

  /// Assignment from nd_elem_access_pxy
  ndarray& operator=(const ndarray_details::nd_elem_access_pxy<ElemType, NDim>& pxy)
  {
    if (Super::m_own) delete [] Super::m_data;
    Super::m_data = pxy.m_data;
    Super::m_own = false;
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

  /// Returns iterator to the beginning of data, iteration is performed in memory order
  iterator begin() { return Super::m_data; }
  const_iterator begin() const { return Super::m_data; }

  /// Returns iterator to the end of data
  iterator end() { return Super::m_data + size(); }
  const_iterator end() const { return Super::m_data + size(); }

  /// Returns reverse iterators
  reverse_iterator rbegin() { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
  reverse_iterator rend() { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

  /**
   *  @brief Make a new copy of the data.
   */
  ndarray copy() const
  {
    size_type dsize = size();
    element* data = new element[dsize];
    std::copy(Super::m_data, Super::m_data+dsize, data);
    return ndarray(data, Super::m_shape, Super::m_strides, true);
  }

protected:

private:

};

#endif // NDARRAY_NDARRAY_H
