#ifndef NDARRAY_ND_ELEM_ACCESS_PXY_H
#define NDARRAY_ND_ELEM_ACCESS_PXY_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class nd_elem_access_pxy.
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
template <typename ElemType, unsigned NDim> class ndarray ;

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
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

template <typename ElemType, unsigned NDim>
class nd_elem_access_pxy  {
public:

  typedef unsigned shape_t;
  typedef int stride_t;

  // Default constructor
  nd_elem_access_pxy (const boost::shared_ptr<ElemType>& data,
                      const shape_t shape[], const stride_t strides[])
    : m_data(data), m_shape(shape), m_strides(strides) {}

  nd_elem_access_pxy<ElemType, NDim-1> operator[](int i) const {
    boost::shared_ptr<ElemType> ptr(m_data, m_data.get() + i*m_strides[0]);
    return nd_elem_access_pxy<ElemType, NDim-1>(ptr, m_shape+1, m_strides+1);
  }

private:

  // ndarray<T,N> can be constructed from this type, needs access to internals
  friend class ndarray<ElemType, NDim>;

  boost::shared_ptr<ElemType> m_data;
  const shape_t* m_shape;
  const stride_t* m_strides;
};

template <typename ElemType>
class nd_elem_access_pxy<ElemType, 1> {
public:

  typedef unsigned shape_t;
  typedef int stride_t;

  // Default constructor
  nd_elem_access_pxy (const boost::shared_ptr<ElemType>& data,
                      const shape_t shape[], const stride_t strides[])
    : m_data(data), m_shape(shape), m_strides(strides) {}

  ElemType& operator[](int i) const { return m_data.get()[i*m_strides[0]]; }

private:

  // ndarray<T,1> can be constructed from this type, needs access to internals
  friend class ndarray<ElemType, 1>;

  boost::shared_ptr<ElemType> m_data;
  const shape_t* m_shape;
  const stride_t* m_strides;
};

} // namespace ndarray_details

#endif // NDARRAY_ND_ELEM_ACCESS_PXY_H
