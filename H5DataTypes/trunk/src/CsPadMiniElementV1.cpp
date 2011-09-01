//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadMiniElementV1...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/CsPadMiniElementV1.h"

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
#include "pdsdata/cspad/Detector.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

CsPadMiniElementV1::CsPadMiniElementV1 ( const XtcType& data )
  : m_data(data)
{
}

hdf5pp::Type
CsPadMiniElementV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
CsPadMiniElementV1::native_type()
{
  return CsPadElementHeader::native_type();
}

hdf5pp::Type
CsPadMiniElementV1::stored_data_type()
{
  hdf5pp::Type baseType = hdf5pp::TypeTraits<int16_t>::native_type() ;

  hsize_t dims[] = { Pds::CsPad::ColumnsPerASIC, Pds::CsPad::MaxRowsPerASIC*2, 2 } ;
  return hdf5pp::ArrayType::arrayType ( baseType, 3, dims );
}

hdf5pp::Type
CsPadMiniElementV1::cmode_data_type()
{
  return hdf5pp::ArrayType::arrayType<float>(2);
}

} // namespace H5DataTypes
