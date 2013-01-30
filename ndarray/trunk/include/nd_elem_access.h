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

/// @addtogroup ndarray_details

/**
 *  @ingroup ndarray_details
 *
 *  @brief Class nd_elem_access which implements element access for ndarray.
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
class nd_elem_access : public nd_data<ElemType, NDim> {

  typedef nd_data<ElemType, NDim> Super;

public:

  nd_elem_access() {}

  template <typename T>
  nd_elem_access(const nd_elem_access<T, NDim>& other)
    : Super(other)
  {
  }

  template <typename T>
  nd_elem_access& operator=(const nd_elem_access<T, NDim>& other) {
    Super::operator=(other);
    return *this;
  }

  nd_elem_access_pxy<ElemType, NDim-1> operator[](int i) const {
    boost::shared_ptr<ElemType> ptr(Super::m_data, Super::m_data.get() + i*Super::m_strides[0]);
    return nd_elem_access_pxy<ElemType, NDim-1>(ptr, Super::m_shape+1, Super::m_strides+1);
  }

protected:

private:
};

/**
 * Specialization for 1-dim arrays for which operator[] returns element
 * itself instead of special proxy object.
 */
template <typename ElemType>
class nd_elem_access<ElemType, 1> : public nd_data<ElemType, 1> {

  typedef nd_data<ElemType, 1> Super;

public:

  nd_elem_access() {}

  template <typename T>
  nd_elem_access(const nd_elem_access<T, 1>& other)
    : Super(other)
  {
  }

  template <typename T>
  nd_elem_access& operator=(const nd_elem_access<T, 1>& other) {
    Super::operator=(other);
    return *this;
  }

  ElemType& operator[](int i) const { return Super::m_data.get()[i*Super::m_strides[0]]; }

protected:

private:

};

} // namespace ndarray_details

#endif // NDARRAY_ND_ELEM_ACCESS_H
