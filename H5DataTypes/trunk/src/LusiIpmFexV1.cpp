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

LusiIpmFexV1::LusiIpmFexV1 ( const XtcType& data )
{
  std::copy( data.channel, data.channel+LusiIpmFexV1_Data::CHSIZE, m_data.channel);
  m_data.sum = data.sum;
  m_data.xpos = data.xpos;
  m_data.ypos = data.ypos;
}

hdf5pp::Type
LusiIpmFexV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
LusiIpmFexV1::native_type()
{
  hdf5pp::ArrayType chType = hdf5pp::ArrayType::arrayType<float>(LusiIpmFexV1_Data::CHSIZE) ;

  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<LusiIpmFexV1_Data>() ;
  type.insert( "channel", offsetof(LusiIpmFexV1_Data, channel), chType ) ;
  type.insert_native<float>( "sum", offsetof(LusiIpmFexV1_Data, sum) ) ;
  type.insert_native<float>( "xpos", offsetof(LusiIpmFexV1_Data, xpos) ) ;
  type.insert_native<float>( "ypos", offsetof(LusiIpmFexV1_Data, ypos) ) ;

  return type ;
}

} // namespace H5DataTypes
