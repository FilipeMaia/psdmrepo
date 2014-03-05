//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpixElementV1...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/EpixElementV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/ArrayType.h"
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/TypeTraits.h"
#include "H5DataTypes/H5DataUtils.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

hdf5pp::Type
EpixElementV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EpixElementV1::native_type()
{
  typedef EpixElementV1 DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("vc", offsetof(DsType, vc), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("lane", offsetof(DsType, lane), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("acqCount", offsetof(DsType, acqCount), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("frameNumber", offsetof(DsType, frameNumber), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("ticks", offsetof(DsType, ticks), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("fiducials", offsetof(DsType, fiducials), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("lastWord", offsetof(DsType, lastWord), hdf5pp::TypeTraits<uint32_t>::native_type());
  return type;
}

hdf5pp::Type
EpixElementV1::frame_data_type(int numberOfRows, int numberOfColumns)
{
  hdf5pp::Type baseType = hdf5pp::TypeTraits<uint16_t>::native_type() ;

  hsize_t dims[] = { numberOfRows, numberOfColumns } ;
  return hdf5pp::ArrayType::arrayType ( baseType, 2, dims );
}

hdf5pp::Type
EpixElementV1::excludedRows_data_type(int lastRowExclusions, int numberOfColumns)
{
  hdf5pp::Type baseType = hdf5pp::TypeTraits<uint16_t>::native_type() ;

  hsize_t dims[] = { lastRowExclusions, numberOfColumns } ;
  return hdf5pp::ArrayType::arrayType ( baseType, 2, dims );
}

hdf5pp::Type
EpixElementV1::temperature_data_type(int nAsics)
{
  return hdf5pp::ArrayType::arrayType<uint16_t>(nAsics);
}

} // namespace H5DataTypes
