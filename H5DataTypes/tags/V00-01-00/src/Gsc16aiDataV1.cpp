//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Gsc16aiDataV1...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/Gsc16aiDataV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/ArrayType.h"
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
Gsc16aiDataV1::Gsc16aiDataV1(const XtcType& data)
{
  std::copy(data._timestamp, data._timestamp+NTimestamps, _timestamp);
}

hdf5pp::Type
Gsc16aiDataV1::stored_type()
{
  return native_type();
}

hdf5pp::Type
Gsc16aiDataV1::native_type()
{
  return hdf5pp::ArrayType::arrayType<uint16_t>(NTimestamps);
}

hdf5pp::Type
Gsc16aiDataV1::stored_data_type(const ConfigXtcType& config)
{
  unsigned size = config.lastChan() - config.firstChan() + 1;
  return hdf5pp::ArrayType::arrayType<uint16_t>(size);
}

} // namespace H5DataTypes
