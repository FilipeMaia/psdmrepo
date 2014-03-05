//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ArraycharDataV1...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/ArraycharDataV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/VlenType.h"
#include "hdf5pp/TypeTraits.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

//----------------
// Constructors --
//----------------
ArraycharDataV1::ArraycharDataV1(const XtcType& xdata)
  : numChars(xdata.numChars())
  , vlen_data(0)
  , data(0)
{
  {
    const __typeof__(xdata.data())& arr = xdata.data();
    vlen_data = arr.size();
    data = static_cast<uint8_t*>(malloc(vlen_data*sizeof(uint8_t)));
    std::copy(arr.begin(), arr.end(), data);
  }
}

ArraycharDataV1::~ArraycharDataV1()
{
  free(this->data);
}

hdf5pp::Type
ArraycharDataV1::stored_type()
{
  return native_type();
}

hdf5pp::Type
ArraycharDataV1::native_type()
{
  typedef ArraycharDataV1 DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("numChars", offsetof(DsType, numChars), hdf5pp::TypeTraits<uint64_t>::stored_type());
  hdf5pp::VlenType _array_type_data = hdf5pp::VlenType::vlenType(hdf5pp::TypeTraits<uint8_t>::stored_type());
  type.insert("data", offsetof(DsType, vlen_data), _array_type_data);
  return type;
}

} // namespace H5DataTypes
