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
#include "SITConfig/SITConfig.h"

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

namespace {
  
  // slow bit count
  unsigned bitCount(uint32_t mask, unsigned maxBits) {
    unsigned res = 0;
    for (  ; maxBits ; -- maxBits ) {
      if ( mask & 1 ) ++ res ;
      mask >>= 1 ;
    }
    return res ;
  }
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

CsPadElementV1::CsPadElementV1 ( const XtcType& data )
{
  m_data.tid = data.tid();
  m_data.seq_count = data.seq_count();
  m_data.ticks = data.ticks();
  m_data.fiducials = data.fiducials();
  m_data.acq_count = data.acq_count();
  for( int i = 0 ; i < CsPadElementV1_Data::SbTempSize ; ++ i ) {
    m_data.sb_temp[i] = data.sb_temp(i);
  }
  m_data.virtual_channel = data.virtual_channel();
  m_data.lane = data.lane();
  m_data.op_code = data.op_code();
  m_data.quad = data.quad();
  m_data.frame_type = data.frame_type();
}

hdf5pp::Type
CsPadElementV1::stored_type( const ConfigXtcType& config )
{
  return native_type(config) ;
}

hdf5pp::Type
CsPadElementV1::native_type( const ConfigXtcType& config )
{
  hdf5pp::ArrayType sb_tempType = hdf5pp::ArrayType::arrayType<uint16_t>(CsPadElementV1_Data::SbTempSize) ;
  
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<CsPadElementV1_Data>() ;
  type.insert_native<uint32_t>( "tid", offsetof(CsPadElementV1_Data, tid) );
  type.insert_native<uint32_t>( "seq_count", offsetof(CsPadElementV1_Data, seq_count) );
  type.insert_native<uint32_t>( "ticks", offsetof(CsPadElementV1_Data, ticks) );
  type.insert_native<uint32_t>( "fiducials", offsetof(CsPadElementV1_Data, fiducials) );
  type.insert_native<uint16_t>( "acq_count", offsetof(CsPadElementV1_Data, acq_count) );
  type.insert( "sb_temp", offsetof(CsPadElementV1_Data, sb_temp), sb_tempType );
  type.insert_native<uint8_t>( "virtual_channel", offsetof(CsPadElementV1_Data, virtual_channel) );
  type.insert_native<uint8_t>( "lane", offsetof(CsPadElementV1_Data, lane) );
  type.insert_native<uint8_t>( "op_code", offsetof(CsPadElementV1_Data, op_code) );
  type.insert_native<uint8_t>( "quad", offsetof(CsPadElementV1_Data, quad) );
  type.insert_native<uint8_t>( "frame_type", offsetof(CsPadElementV1_Data, frame_type) );

  const unsigned nElem = ::bitCount(config.quadMask(), Pds::CsPad::MaxQuadsPerSensor);
  return hdf5pp::ArrayType::arrayType ( type, nElem );
}

hdf5pp::Type
CsPadElementV1::stored_data_type( const ConfigXtcType& config )
{
  hdf5pp::Type baseType = hdf5pp::TypeTraits<uint16_t>::native_type() ;

  // get few constants
  const unsigned nElem = ::bitCount(config.quadMask(), Pds::CsPad::MaxQuadsPerSensor);
  const unsigned nAsic = config.numAsicsRead();

  hsize_t dims[] = { nElem, nAsic, Pds::CsPad::MaxRowsPerASIC, Pds::CsPad::ColumnsPerASIC } ;
  return hdf5pp::ArrayType::arrayType ( baseType, 4, dims );
}

} // namespace H5DataTypes
