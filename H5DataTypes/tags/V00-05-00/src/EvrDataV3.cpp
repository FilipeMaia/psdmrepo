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
#include "hdf5pp/VlenType.h"
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
  : numFifoEvents(data.numFifoEvents())
  , fifoEvents(data.fifoEvents().data())
{
}

hdf5pp::Type
EvrDataV3::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EvrDataV3::native_type()
{
  hdf5pp::CompoundType baseType = hdf5pp::CompoundType::compoundType<Pds::EvrData::FIFOEvent>() ;
  baseType.insert_native<uint32_t>( "timestampHigh", 0 ) ;
  baseType.insert_native<uint32_t>( "timestampLow", 4 ) ;
  baseType.insert_native<uint32_t>( "eventCode", 8 ) ;
  hdf5pp::Type fifoType = hdf5pp::VlenType::vlenType ( baseType );

  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<EvrDataV3>() ;
  type.insert( "fifoEvents", 0, fifoType ) ;

  return type;
}

} // namespace H5DataTypes
