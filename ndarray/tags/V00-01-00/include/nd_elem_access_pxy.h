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

  // Default constructor
  nd_elem_access_pxy (const ElemType* data, const unsigned shape[], const unsigned strides[])
    : m_data(data), m_shape(shape), m_strides(strides) {}

  nd_elem_access_pxy<ElemType, NDim-1> operator[](int i) const {
    return nd_elem_access_pxy<ElemType, NDim-1>(m_data + i*m_strides[0], m_shape+1, m_strides+1);
  }

private:

  // ndarray<T,N> can be constructed from this type, needs access to internals
  friend class ndarray<ElemType, NDim>;

  const ElemType* m_data;
  const unsigned* m_shape;
  const unsigned* m_strides;
};

template <typename ElemType>
class nd_elem_access_pxy<ElemType, 1> {
public:

  // Default constructor
  nd_elem_access_pxy (const ElemType* data, const unsigned shape[], const unsigned strides[])
    : m_data(data), m_shape(shape), m_strides(strides) {}

  const ElemType& operator[](int i) const { return m_data[i*m_strides[0]]; }

private:

  // ndarray<T,1> can be constructed from this type, needs access to internals
  friend class ndarray<ElemType, 1>;

  const ElemType* m_data;
  const unsigned* m_shape;
  const unsigned* m_strides;
};

} // namespace ndarray_details

#endif // NDARRAY_ND_ELEM_ACCESS_PXY_H
