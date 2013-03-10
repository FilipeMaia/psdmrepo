#ifndef NDARRAY_FORMAT_H
#define NDARRAY_FORMAT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class format.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <iosfwd>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ndarray/ndarray.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------


namespace ndarray_details {

template <typename ElemType, unsigned NDim>
struct dump_ndarray_data {
  static void dump(std::ostream& str, const ndarray<ElemType, NDim>& array, unsigned offset=0)
  {
    char sep[] = ",\n                 ";
    if (offset+3 < sizeof sep) sep[offset+3] = '\0';
    str << '[';
    unsigned size = array.shape()[0];
    if (size > 7) {
      for (unsigned i = 0; i != 3; ++ i) {
        if (i) str << sep;
        dump_ndarray_data<ElemType, NDim-1>::dump(str, array[i], offset+1);
      }
      str << sep << "...";
      for (unsigned i = size-3; i != size; ++ i) {
        str << sep;
        dump_ndarray_data<ElemType, NDim-1>::dump(str, array[i], offset+1);
      }
    } else {
      for (unsigned i = 0; i != size; ++ i) {
        if (i) str << sep;
        dump_ndarray_data<ElemType, NDim-1>::dump(str, array[i], offset+1);
      }
    }
    str << ']';
  }
};

template <typename ElemType>
struct dump_ndarray_data<ElemType, 1> {
  static void dump(std::ostream& str, const ndarray<ElemType, 1>& array, unsigned offset=0)
  {
    str << '[';
    unsigned size = array.size();
    if (size > 9) {
      for (unsigned i = 0; i != 4; ++ i) str << array[i] << ", ";
      str << "...";
      for (unsigned i = size-4; i != size; ++ i) str << ", " << array[i];
    } else {
      for (unsigned i = 0; i != size; ++ i) {
        if (i) str << ", ";
        str << array[i];
      }
    }
    str << ']';
  }
};

} // namespace ndarray_details

/// @addtogroup ndarray

/**
 *  @ingroup ndarray
 *
 *  @brief Methods for formatting ndarray objects.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

template <typename ElemType, unsigned NDim>
std::ostream&
operator<<(std::ostream& str, const ndarray<ElemType, NDim>& array)
{
  str << "ndarray<T>(shape=(";
  for (unsigned i = 0; i != NDim; ++ i) {
    if (i) str << ',';
    str << array.shape()[i];
  }
  str << "), data=";
  ndarray_details::dump_ndarray_data<ElemType, NDim>::dump(str, array);
  str << ")";
  return str;
}


#endif // NDARRAY_FORMAT_H
