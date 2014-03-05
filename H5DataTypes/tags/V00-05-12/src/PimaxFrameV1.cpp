//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PimaxFrameV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/PimaxFrameV1.h"

//-----------------
// C/C++ Headers --
//-----------------

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
PimaxFrameV1::PimaxFrameV1 ( const XtcType& frame )
  : shotIdStart(frame.shotIdStart())
  , readoutTime(frame.readoutTime())
  , temperature(frame.temperature())
{
}

hdf5pp::Type
PimaxFrameV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
PimaxFrameV1::native_type()
{
  typedef PimaxFrameV1 DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("shotIdStart", offsetof(DsType, shotIdStart), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("readoutTime", offsetof(DsType, readoutTime), hdf5pp::TypeTraits<float>::native_type());
  type.insert("temperature", offsetof(DsType, temperature), hdf5pp::TypeTraits<float>::native_type());
  return type;
}

hdf5pp::Type
PimaxFrameV1::stored_data_type(uint32_t height, uint32_t width)
{
  hdf5pp::Type baseType = hdf5pp::TypeTraits<uint16_t>::native_type() ;

  hsize_t dims[] = { height, width } ;
  return hdf5pp::ArrayType::arrayType ( baseType, 2, dims );
}

} // namespace H5DataTypes
