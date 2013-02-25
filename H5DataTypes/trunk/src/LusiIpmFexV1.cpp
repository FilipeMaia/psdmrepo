//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class LusiIpmFexV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/LusiIpmFexV1.h"

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

LusiIpmFexV1::LusiIpmFexV1 ( const XtcType& data )
  : sum(data.sum)
  , xpos(data.xpos)
  , ypos(data.ypos)
{
  std::copy(data.channel, data.channel+LusiIpmFexV1::CHSIZE, channel);
}

hdf5pp::Type
LusiIpmFexV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
LusiIpmFexV1::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<LusiIpmFexV1>() ;
  type.insert_native<float>( "channel", offsetof(LusiIpmFexV1, channel), LusiIpmFexV1::CHSIZE ) ;
  type.insert_native<float>( "sum", offsetof(LusiIpmFexV1, sum) ) ;
  type.insert_native<float>( "xpos", offsetof(LusiIpmFexV1, xpos) ) ;
  type.insert_native<float>( "ypos", offsetof(LusiIpmFexV1, ypos) ) ;

  return type ;
}

} // namespace H5DataTypes
