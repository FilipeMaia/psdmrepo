//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrDataV3...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "SITConfig/SITConfig.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/EvrDataV3.h"

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
EvrDataV3::EvrDataV3 ( const XtcType& data )
{
  m_data.numFifoEvents = data.numFifoEvents() ;
}

hdf5pp::Type
EvrDataV3::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EvrDataV3::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<EvrDataV3>() ;
  type.insert_native<uint32_t>( "numFifoEvents", offsetof(EvrDataV3_Data,numFifoEvents) ) ;

  return type;
}

hdf5pp::Type
EvrDataV3::stored_fifoevent_type(const ConfigXtcType& config)
{
  hdf5pp::CompoundType baseType = hdf5pp::CompoundType::compoundType<Pds::EvrData::DataV3::FIFOEvent>() ;
  baseType.insert_native<uint32_t>( "timestampHigh", offsetof(Pds::EvrData::DataV3::FIFOEvent,TimestampHigh) ) ;
  baseType.insert_native<uint32_t>( "timestampLow", offsetof(Pds::EvrData::DataV3::FIFOEvent,TimestampLow) ) ;
  baseType.insert_native<uint32_t>( "eventCode", offsetof(Pds::EvrData::DataV3::FIFOEvent,EventCode) ) ;

  hsize_t dims[] = { config.neventcodes() } ;
  return hdf5pp::ArrayType::arrayType ( baseType, 1, dims );
}

} // namespace H5DataTypes
