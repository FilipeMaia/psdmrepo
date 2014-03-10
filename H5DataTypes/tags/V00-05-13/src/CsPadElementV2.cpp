//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadElementV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/CsPadElementV2.h"

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

CsPadElementV2::CsPadElementV2 ( const XtcType& data )
  : tid(data.tid())
  , seq_count(data.seq_count())
  , ticks(data.ticks())
  , fiducials(data.fiducials())
  , acq_count(data.acq_count())
  , virtual_channel(data.virtual_channel())
  , lane(data.lane())
  , op_code(data.op_code())
  , quad(data.quad())
  , frame_type(data.frame_type())
{
  const ndarray<const uint16_t, 1>& sb_temp = data.sb_temp();
  std::copy(sb_temp.begin(), sb_temp.end(), this->sb_temp);
}

hdf5pp::Type
CsPadElementV2::stored_type(unsigned nQuad)
{
  return native_type(nQuad) ;
}

hdf5pp::Type
CsPadElementV2::native_type(unsigned nQuad)
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<CsPadElementV2>() ;
  type.insert_native<uint32_t>( "tid", offsetof(CsPadElementV2, tid) );
  type.insert_native<uint32_t>( "seq_count", offsetof(CsPadElementV2, seq_count) );
  type.insert_native<uint32_t>( "ticks", offsetof(CsPadElementV2, ticks) );
  type.insert_native<uint32_t>( "fiducials", offsetof(CsPadElementV2, fiducials) );
  type.insert_native<uint16_t>( "acq_count", offsetof(CsPadElementV2, acq_count) );
  type.insert_native<uint16_t>( "sb_temp", offsetof(CsPadElementV2, sb_temp), SbTempSize );
  type.insert_native<uint8_t>( "virtual_channel", offsetof(CsPadElementV2, virtual_channel) );
  type.insert_native<uint8_t>( "lane", offsetof(CsPadElementV2, lane) );
  type.insert_native<uint8_t>( "op_code", offsetof(CsPadElementV2, op_code) );
  type.insert_native<uint8_t>( "quad", offsetof(CsPadElementV2, quad) );
  type.insert_native<uint8_t>( "frame_type", offsetof(CsPadElementV2, frame_type) );

  return hdf5pp::ArrayType::arrayType(type, nQuad);
}

hdf5pp::Type
CsPadElementV2::stored_data_type(unsigned nSect)
{
  hdf5pp::Type baseType = hdf5pp::TypeTraits<int16_t>::native_type() ;

  hsize_t dims[] = { nSect, Pds::CsPad::ColumnsPerASIC, Pds::CsPad::MaxRowsPerASIC*2 } ;
  return hdf5pp::ArrayType::arrayType ( baseType, 3, dims );
}

hdf5pp::Type
CsPadElementV2::cmode_data_type(unsigned nSect)
{
  return hdf5pp::ArrayType::arrayType<float> ( nSect );
}

} // namespace H5DataTypes
