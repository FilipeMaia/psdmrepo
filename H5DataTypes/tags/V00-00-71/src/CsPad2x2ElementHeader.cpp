//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2ElementHeader...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/CsPad2x2ElementHeader.h"

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

//----------------
// Constructors --
//----------------
CsPad2x2ElementHeader::CsPad2x2ElementHeader(const XtcType& data)
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
    for( int i = 0 ; i < SbTempSize ; ++ i ) {
      sb_temp[i] = data.sb_temp(i);
    }
}

//--------------
// Destructor --
//--------------
CsPad2x2ElementHeader::~CsPad2x2ElementHeader ()
{
}

hdf5pp::Type
CsPad2x2ElementHeader::native_type()
{
  hdf5pp::ArrayType sb_tempType = hdf5pp::ArrayType::arrayType<uint16_t>(SbTempSize) ;

  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<CsPad2x2ElementHeader>() ;
  type.insert_native<uint32_t>( "tid", offsetof(CsPad2x2ElementHeader, tid) );
  type.insert_native<uint32_t>( "seq_count", offsetof(CsPad2x2ElementHeader, seq_count) );
  type.insert_native<uint32_t>( "ticks", offsetof(CsPad2x2ElementHeader, ticks) );
  type.insert_native<uint32_t>( "fiducials", offsetof(CsPad2x2ElementHeader, fiducials) );
  type.insert_native<uint16_t>( "acq_count", offsetof(CsPad2x2ElementHeader, acq_count) );
  type.insert( "sb_temp", offsetof(CsPad2x2ElementHeader, sb_temp), sb_tempType );
  type.insert_native<uint8_t>( "virtual_channel", offsetof(CsPad2x2ElementHeader, virtual_channel) );
  type.insert_native<uint8_t>( "lane", offsetof(CsPad2x2ElementHeader, lane) );
  type.insert_native<uint8_t>( "op_code", offsetof(CsPad2x2ElementHeader, op_code) );
  type.insert_native<uint8_t>( "quad", offsetof(CsPad2x2ElementHeader, quad) );
  type.insert_native<uint8_t>( "frame_type", offsetof(CsPad2x2ElementHeader, frame_type) );

  return type;
}

} // namespace H5DataTypes
