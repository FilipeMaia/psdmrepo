//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PrincetonInfoV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/PrincetonInfoV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/TypeTraits.h"
#include "H5DataTypes/H5DataUtils.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

PrincetonInfoV1::PrincetonInfoV1 ( const XtcType& data )
{
  m_data.temperature = data.temperature();
}

hdf5pp::Type
PrincetonInfoV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
PrincetonInfoV1::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<PrincetonInfoV1>() ;
  confType.insert_native<float>( "temperature", offsetof(PrincetonInfoV1_Data,temperature) );

  return confType ;
}

} // namespace H5DataTypes
