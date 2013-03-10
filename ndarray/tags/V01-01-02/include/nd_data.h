#ifndef NDARRAY_ND_DATA_H
#define NDARRAY_ND_DATA_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class nd_data.
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

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ndarray_details {

/// @addtogroup ndarray_details

/**
 *  @ingroup ndarray_details
 *
 *  @brief C++ source file code template.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

template <typename ElemType, unsigned NDim>
struct nd_data  {

  typedef unsigned shape_t;
  typedef int stride_t;

  /// Default constructor
  nd_data()
  {
    std::fill_n(m_shape, NDim, 0U);
    std::fill_n(m_strides, NDim, 0U);
  }

  /**
   *  "Copy" constructor from compatible type
   */
  template <typename T>
  nd_data(const nd_data<T, NDim>& other)
    : m_data(other.m_data)
  {
    std::copy(other.m_shape, other.m_shape+NDim, m_shape);
    std::copy(other.m_strides, other.m_strides+NDim, m_strides);
  }

  /**
   *  Assignment from same type
   */
  nd_data& operator=(const nd_data& other)
  {
    if (this != &other) {
      m_data = other.m_data;
      std::copy(other.m_shape, other.m_shape+NDim, m_shape);
      std::copy(other.m_strides, other.m_strides+NDim, m_strides);
    }
    return *this;
  }

  /**
   *  Assignment from compatible type
   */
  template <typename T>
  nd_data& operator=(const nd_data<T, NDim>& other)
  {
    m_data = other.m_data;
    std::copy(other.m_shape, other.m_shape+NDim, m_shape);
    std::copy(other.m_strides, other.m_strides+NDim, m_strides);
    return *this;
  }

  /**
   *  Return shared pointer to contained data.
   *
   *  This could be used in rare cases when it's necessary to provide
   *  access to data without loosing ownership.
   */
  boost::shared_ptr<ElemType> data_ptr() const { return m_data; }



  boost::shared_ptr<ElemType> m_data;          ///< Pointer to the data array
  shape_t m_shape[NDim];    ///< Array dimensions
  stride_t m_strides[NDim];  ///< Array strides
};

} // namespace ndarray_details

#endif // NDARRAY_ND_DATA_H
