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
  for( int i = 0 ; i < CsPadElementHeader_Data::SbTempSize ; ++ i ) {
    m_data.sb_temp[i] = data.sb_temp(i);
  }
  m_data.virtual_channel = data.virtual_channel();
  m_data.lane = data.lane();
  m_data.op_code = data.op_code();
  m_data.quad = data.quad();
  m_data.frame_type = data.frame_type();
}

hdf5pp::Type
CsPadElementV1::stored_type(unsigned nQuad)
{
  return native_type(nQuad) ;
}

hdf5pp::Type
CsPadElementV1::native_type(unsigned nQuad)
{
  hdf5pp::ArrayType sb_tempType = hdf5pp::ArrayType::arrayType<uint16_t>(CsPadElementHeader_Data::SbTempSize) ;
  
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<CsPadElementHeader_Data>() ;
  type.insert_native<uint32_t>( "tid", offsetof(CsPadElementHeader_Data, tid) );
  type.insert_native<uint32_t>( "seq_count", offsetof(CsPadElementHeader_Data, seq_count) );
  type.insert_native<uint32_t>( "ticks", offsetof(CsPadElementHeader_Data, ticks) );
  type.insert_native<uint32_t>( "fiducials", offsetof(CsPadElementHeader_Data, fiducials) );
  type.insert_native<uint16_t>( "acq_count", offsetof(CsPadElementHeader_Data, acq_count) );
  type.insert( "sb_temp", offsetof(CsPadElementHeader_Data, sb_temp), sb_tempType );
  type.insert_native<uint8_t>( "virtual_channel", offsetof(CsPadElementHeader_Data, virtual_channel) );
  type.insert_native<uint8_t>( "lane", offsetof(CsPadElementHeader_Data, lane) );
  type.insert_native<uint8_t>( "op_code", offsetof(CsPadElementHeader_Data, op_code) );
  type.insert_native<uint8_t>( "quad", offsetof(CsPadElementHeader_Data, quad) );
  type.insert_native<uint8_t>( "frame_type", offsetof(CsPadElementHeader_Data, frame_type) );

  return hdf5pp::ArrayType::arrayType ( type, nQuad );
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
