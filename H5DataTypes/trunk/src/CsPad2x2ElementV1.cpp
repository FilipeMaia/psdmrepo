//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2ElementV1...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/CsPad2x2ElementV1.h"

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

CsPad2x2ElementV1::CsPad2x2ElementV1 ( const XtcType& data )
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
CsPad2x2ElementV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
CsPad2x2ElementV1::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<CsPad2x2ElementV1>() ;
  type.insert_native<uint32_t>( "tid", offsetof(CsPad2x2ElementV1, tid) );
  type.insert_native<uint32_t>( "seq_count", offsetof(CsPad2x2ElementV1, seq_count) );
  type.insert_native<uint32_t>( "ticks", offsetof(CsPad2x2ElementV1, ticks) );
  type.insert_native<uint32_t>( "fiducials", offsetof(CsPad2x2ElementV1, fiducials) );
  type.insert_native<uint16_t>( "acq_count", offsetof(CsPad2x2ElementV1, acq_count) );
  type.insert_native<uint16_t>( "sb_temp", offsetof(CsPad2x2ElementV1, sb_temp), SbTempSize );
  type.insert_native<uint8_t>( "virtual_channel", offsetof(CsPad2x2ElementV1, virtual_channel) );
  type.insert_native<uint8_t>( "lane", offsetof(CsPad2x2ElementV1, lane) );
  type.insert_native<uint8_t>( "op_code", offsetof(CsPad2x2ElementV1, op_code) );
  type.insert_native<uint8_t>( "quad", offsetof(CsPad2x2ElementV1, quad) );
  type.insert_native<uint8_t>( "frame_type", offsetof(CsPad2x2ElementV1, frame_type) );

  return type;
}

hdf5pp::Type
CsPad2x2ElementV1::stored_data_type()
{
  hdf5pp::Type baseType = hdf5pp::TypeTraits<int16_t>::native_type() ;

  hsize_t dims[] = { Pds::CsPad2x2::ColumnsPerASIC, Pds::CsPad2x2::MaxRowsPerASIC*2, 2 } ;
  return hdf5pp::ArrayType::arrayType ( baseType, 3, dims );
}

hdf5pp::Type
CsPad2x2ElementV1::cmode_data_type()
{
  return hdf5pp::ArrayType::arrayType<float>(2);
}

} // namespace H5DataTypes
