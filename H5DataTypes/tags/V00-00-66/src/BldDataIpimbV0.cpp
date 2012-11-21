//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataIpimbV0...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/BldDataIpimbV0.h"

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

BldDataIpimbV0::BldDataIpimbV0 ( const XtcType& data )
  : ipimbData(data.ipimbData)
  , ipimbConfig(data.ipimbConfig)
  , ipmFexData(data.ipmFexData)
{
}

BldDataIpimbV0::~BldDataIpimbV0 ()
{
}

hdf5pp::Type
BldDataIpimbV0::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
BldDataIpimbV0::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<BldDataIpimbV0>() ;

  type.insert( "ipimbData", offsetof(BldDataIpimbV0,ipimbData), IpimbDataV1::native_type() );
  type.insert( "ipimbConfig", offsetof(BldDataIpimbV0,ipimbConfig), IpimbConfigV1::native_type() );
  type.insert( "ipmFexData", offsetof(BldDataIpimbV0,ipmFexData), LusiIpmFexV1::native_type() );

  return type ;
}

} // namespace H5DataTypes
