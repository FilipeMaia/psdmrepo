//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadElementHeader...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/CsPadElementHeader.h"

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
CsPadElementHeader::CsPadElementHeader(const XtcType& data)
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
CsPadElementHeader::~CsPadElementHeader ()
{
}

hdf5pp::Type
CsPadElementHeader::native_type()
{
  hdf5pp::ArrayType sb_tempType = hdf5pp::ArrayType::arrayType<uint16_t>(SbTempSize) ;

  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<CsPadElementHeader>() ;
  type.insert_native<uint32_t>( "tid", offsetof(CsPadElementHeader, tid) );
  type.insert_native<uint32_t>( "seq_count", offsetof(CsPadElementHeader, seq_count) );
  type.insert_native<uint32_t>( "ticks", offsetof(CsPadElementHeader, ticks) );
  type.insert_native<uint32_t>( "fiducials", offsetof(CsPadElementHeader, fiducials) );
  type.insert_native<uint16_t>( "acq_count", offsetof(CsPadElementHeader, acq_count) );
  type.insert( "sb_temp", offsetof(CsPadElementHeader, sb_temp), sb_tempType );
  type.insert_native<uint8_t>( "virtual_channel", offsetof(CsPadElementHeader, virtual_channel) );
  type.insert_native<uint8_t>( "lane", offsetof(CsPadElementHeader, lane) );
  type.insert_native<uint8_t>( "op_code", offsetof(CsPadElementHeader, op_code) );
  type.insert_native<uint8_t>( "quad", offsetof(CsPadElementHeader, quad) );
  type.insert_native<uint8_t>( "frame_type", offsetof(CsPadElementHeader, frame_type) );

  return type;
}

} // namespace H5DataTypes
