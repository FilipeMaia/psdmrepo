//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpicsPvHeader...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/EpicsPvHeader.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"

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
EpicsPvHeader::EpicsPvHeader ( const XtcType& xtc )
  : pvId(xtc.iPvId)
  , dbrType(xtc.iDbrType)
  , numElements(xtc.iNumElements)
{
}

//--------------
// Destructor --
//--------------
EpicsPvHeader::~EpicsPvHeader ()
{
}

hdf5pp::Type
EpicsPvHeader::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EpicsPvHeader::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<EpicsPvHeader>() ;
  type.insert_native<int16_t>( "pvId", offsetof(EpicsPvHeader,pvId) ) ;
  type.insert_native<int16_t>( "dbrType", offsetof(EpicsPvHeader,dbrType) ) ;
  type.insert_native<int16_t>( "numElements", offsetof(EpicsPvHeader,numElements) ) ;

  return type ;
}

} // namespace H5DataTypes
