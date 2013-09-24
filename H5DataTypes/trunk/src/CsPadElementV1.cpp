//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadElementV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/CsPadElementV1.h"

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


CsPadElementV1::CsPadElementV1 ( const XtcType& data )
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
CsPadElementV1::stored_type(unsigned nQuad)
{
  return native_type(nQuad) ;
}

hdf5pp::Type
CsPadElementV1::native_type(unsigned nQuad)
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<CsPadElementV1>() ;
  type.insert_native<uint32_t>( "tid", offsetof(CsPadElementV1, tid) );
  type.insert_native<uint32_t>( "seq_count", offsetof(CsPadElementV1, seq_count) );
  type.insert_native<uint32_t>( "ticks", offsetof(CsPadElementV1, ticks) );
  type.insert_native<uint32_t>( "fiducials", offsetof(CsPadElementV1, fiducials) );
  type.insert_native<uint16_t>( "acq_count", offsetof(CsPadElementV1, acq_count) );
  type.insert_native<uint16_t>( "sb_temp", offsetof(CsPadElementV1, sb_temp), SbTempSize );
  type.insert_native<uint8_t>( "virtual_channel", offsetof(CsPadElementV1, virtual_channel) );
  type.insert_native<uint8_t>( "lane", offsetof(CsPadElementV1, lane) );
  type.insert_native<uint8_t>( "op_code", offsetof(CsPadElementV1, op_code) );
  type.insert_native<uint8_t>( "quad", offsetof(CsPadElementV1, quad) );
  type.insert_native<uint8_t>( "frame_type", offsetof(CsPadElementV1, frame_type) );

  return hdf5pp::ArrayType::arrayType(type, nQuad);
}

hdf5pp::Type
CsPadElementV1::stored_data_type(unsigned nQuad, unsigned nSect)
{
  hdf5pp::Type baseType = hdf5pp::TypeTraits<int16_t>::native_type() ;

  hsize_t dims[] = { nQuad, nSect, Pds::CsPad::ColumnsPerASIC, Pds::CsPad::MaxRowsPerASIC*2 } ;
  return hdf5pp::ArrayType::arrayType ( baseType, 4, dims );
}

hdf5pp::Type
CsPadElementV1::cmode_data_type(unsigned nQuad, unsigned nSect)
{
  hdf5pp::Type baseType = hdf5pp::TypeTraits<float>::native_type() ;

  hsize_t dims[] = { nQuad, nSect } ;
  return hdf5pp::ArrayType::arrayType ( baseType, 2, dims );
}

} // namespace H5DataTypes
