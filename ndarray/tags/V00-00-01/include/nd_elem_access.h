#ifndef NDARRAY_ND_ELEM_ACCESS_H
#define NDARRAY_ND_ELEM_ACCESS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class nd_elem_access.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "ndarray/nd_data.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ndarray/nd_elem_access_pxy.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ndarray_details {

/// @addtogroup ndarray

/**
 *  @ingroup ndarray
 *
 *  @brief C++ source file code template.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

template <typename ElemType, unsigned NDim>
class nd_elem_access : protected nd_data<ElemType, NDim> {

    typedef nd_data<ElemType, NDim> Super;

public:

  nd_elem_access_pxy<ElemType, NDim-1> operator[](int i) {
    return nd_elem_access_pxy<ElemType, NDim-1>(Super::m_data + i*Super::m_strides[0], Super::m_shape+1, Super::m_strides+1);
  }
  const nd_elem_access_pxy<ElemType, NDim-1> operator[](int i) const {
    ElemType* data = const_cast<ElemType*>(Super::m_data);
    return nd_elem_access_pxy<ElemType, NDim-1>(data + i*Super::m_strides[0], Super::m_shape+1, Super::m_strides+1);
  }

protected:

private:
};

/**
 * Specialization for 1-dim arrays for which operator[] returns element
 * itself instead of special proxy object.
 */
template <typename ElemType>
class nd_elem_access<ElemType, 1> : protected nd_data<ElemType, 1> {

  typedef nd_data<ElemType, 1> Super;

public:

  ElemType& operator[](int i) { return Super::m_data[i*Super::m_strides[0]]; }
  const ElemType& operator[](int i) const { return Super::m_data[i*Super::m_strides[0]]; }

protected:

private:

};

} // namespace ndarray_details

#endif // NDARRAY_ND_ELEM_ACCESS_H
