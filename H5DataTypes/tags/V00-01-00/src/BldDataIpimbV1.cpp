//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataIpimbV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/BldDataIpimbV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/TypeTraits.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

BldDataIpimbV1::BldDataIpimbV1 ( const XtcType& data )
  : ipimbData(data.ipimbData)
  , ipimbConfig(data.ipimbConfig)
  , ipmFexData(data.ipmFexData)
{
}

BldDataIpimbV1::~BldDataIpimbV1 ()
{
}

hdf5pp::Type
BldDataIpimbV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
BldDataIpimbV1::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<BldDataIpimbV1>() ;

  type.insert( "ipimbData", offsetof(BldDataIpimbV1,ipimbData), IpimbDataV2::native_type() );
  type.insert( "ipimbConfig", offsetof(BldDataIpimbV1,ipimbConfig), IpimbConfigV2::native_type() );
  type.insert( "ipmFexData", offsetof(BldDataIpimbV1,ipmFexData), LusiIpmFexV1::native_type() );

  return type ;
}

} // namespace H5DataTypes
