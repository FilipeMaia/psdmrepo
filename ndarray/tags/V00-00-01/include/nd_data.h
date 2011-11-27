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

/// @addtogroup ndarray

/**
 *  @ingroup ndarray
 *
 *  @brief C++ source file code template.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

template <typename ElemType, unsigned NDimT>
struct nd_data  {

  enum { NDim = NDimT };

  // Default constructor
  nd_data () {}

  // Destructor
  ~nd_data () {}

  ElemType* m_data;          ///< Pointer to the data array
  unsigned m_shape[NDim];    ///< Array dimensions
  unsigned m_strides[NDim];  ///< Array strides
  bool m_own;                ///< If value is true then m_data array will be deleted in destructor
};

} // namespace ndarray_details

#endif // NDARRAY_ND_DATA_H
