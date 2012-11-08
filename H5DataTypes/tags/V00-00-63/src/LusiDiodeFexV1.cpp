//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class LusiDiodeFexV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/LusiDiodeFexV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
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

LusiDiodeFexV1::LusiDiodeFexV1 ( const XtcType& data )
{
  m_data.value = data.value;
}

hdf5pp::Type
LusiDiodeFexV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
LusiDiodeFexV1::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<LusiDiodeFexV1_Data>() ;
  type.insert_native<float>( "value", offsetof(LusiDiodeFexV1_Data, value) ) ;

  return type ;
}

} // namespace H5DataTypes
