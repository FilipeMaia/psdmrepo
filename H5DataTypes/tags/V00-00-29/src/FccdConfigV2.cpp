//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FccdConfigV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "SITConfig/SITConfig.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/FccdConfigV2.h"

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

FccdConfigV2::FccdConfigV2 ( const Pds::FCCD::FccdConfigV2& data )
{
  m_data.width = data.width();
  m_data.height = data.height();
  m_data.trimmedWidth = data.trimmedWidth();
  m_data.trimmedHeight = data.trimmedHeight();
  m_data.outputMode = data.outputMode();
  m_data.ccdEnable = data.ccdEnable();
  m_data.focusMode = data.focusMode();
  m_data.exposureTime = data.exposureTime();
  m_data.dacVoltage1 = data.dacVoltage1();
  m_data.dacVoltage2 = data.dacVoltage2();
  m_data.dacVoltage3 = data.dacVoltage3();
  m_data.dacVoltage4 = data.dacVoltage4();
  m_data.dacVoltage5 = data.dacVoltage5();
  m_data.dacVoltage6 = data.dacVoltage6();
  m_data.dacVoltage7 = data.dacVoltage7();
  m_data.dacVoltage8 = data.dacVoltage8();
  m_data.dacVoltage9 = data.dacVoltage9();
  m_data.dacVoltage10 = data.dacVoltage10();
  m_data.dacVoltage11 = data.dacVoltage11();
  m_data.dacVoltage12 = data.dacVoltage12();
  m_data.dacVoltage13 = data.dacVoltage13();
  m_data.dacVoltage14 = data.dacVoltage14();
  m_data.dacVoltage15 = data.dacVoltage15();
  m_data.dacVoltage16 = data.dacVoltage16();
  m_data.dacVoltage17 = data.dacVoltage17();
  m_data.waveform0 = data.waveform0();
  m_data.waveform1 = data.waveform1();
  m_data.waveform2 = data.waveform2();
  m_data.waveform3 = data.waveform3();
  m_data.waveform4 = data.waveform4();
  m_data.waveform5 = data.waveform5();
  m_data.waveform6 = data.waveform6();
  m_data.waveform7 = data.waveform7();
  m_data.waveform8 = data.waveform8();
  m_data.waveform9 = data.waveform9();
  m_data.waveform10 = data.waveform10();
  m_data.waveform11 = data.waveform11();
  m_data.waveform12 = data.waveform12();
  m_data.waveform13 = data.waveform13();
  m_data.waveform14 = data.waveform14();
}

hdf5pp::Type
FccdConfigV2::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
FccdConfigV2::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<FccdConfigV2_Data>() ;
  confType.insert_native<uint32_t>( "width", offsetof(FccdConfigV2_Data,width) );
  confType.insert_native<uint32_t>( "height", offsetof(FccdConfigV2_Data,height) );
  confType.insert_native<uint32_t>( "trimmedWidth", offsetof(FccdConfigV2_Data,trimmedWidth) );
  confType.insert_native<uint32_t>( "trimmedHeight", offsetof(FccdConfigV2_Data,trimmedHeight) );
  confType.insert_native<uint16_t>( "outputMode", offsetof(FccdConfigV2_Data,outputMode) );
  confType.insert_native<uint8_t>( "ccdEnable", offsetof(FccdConfigV2_Data,ccdEnable) );
  confType.insert_native<uint8_t>( "focusMode", offsetof(FccdConfigV2_Data,focusMode) );
  confType.insert_native<uint32_t>( "exposureTime", offsetof(FccdConfigV2_Data,exposureTime) );
  confType.insert_native<float>( "dacVoltage1", offsetof(FccdConfigV2_Data,dacVoltage1) );
  confType.insert_native<float>( "dacVoltage2", offsetof(FccdConfigV2_Data,dacVoltage2) );
  confType.insert_native<float>( "dacVoltage3", offsetof(FccdConfigV2_Data,dacVoltage3) );
  confType.insert_native<float>( "dacVoltage4", offsetof(FccdConfigV2_Data,dacVoltage4) );
  confType.insert_native<float>( "dacVoltage5", offsetof(FccdConfigV2_Data,dacVoltage5) );
  confType.insert_native<float>( "dacVoltage6", offsetof(FccdConfigV2_Data,dacVoltage6) );
  confType.insert_native<float>( "dacVoltage7", offsetof(FccdConfigV2_Data,dacVoltage7) );
  confType.insert_native<float>( "dacVoltage8", offsetof(FccdConfigV2_Data,dacVoltage8) );
  confType.insert_native<float>( "dacVoltage9", offsetof(FccdConfigV2_Data,dacVoltage9) );
  confType.insert_native<float>( "dacVoltage10", offsetof(FccdConfigV2_Data,dacVoltage10) );
  confType.insert_native<float>( "dacVoltage11", offsetof(FccdConfigV2_Data,dacVoltage11) );
  confType.insert_native<float>( "dacVoltage12", offsetof(FccdConfigV2_Data,dacVoltage12) );
  confType.insert_native<float>( "dacVoltage13", offsetof(FccdConfigV2_Data,dacVoltage13) );
  confType.insert_native<float>( "dacVoltage14", offsetof(FccdConfigV2_Data,dacVoltage14) );
  confType.insert_native<float>( "dacVoltage15", offsetof(FccdConfigV2_Data,dacVoltage15) );
  confType.insert_native<float>( "dacVoltage16", offsetof(FccdConfigV2_Data,dacVoltage16) );
  confType.insert_native<float>( "dacVoltage17", offsetof(FccdConfigV2_Data,dacVoltage17) );
  confType.insert_native<uint16_t>( "waveform0", offsetof(FccdConfigV2_Data,waveform0) );
  confType.insert_native<uint16_t>( "waveform1", offsetof(FccdConfigV2_Data,waveform1) );
  confType.insert_native<uint16_t>( "waveform2", offsetof(FccdConfigV2_Data,waveform2) );
  confType.insert_native<uint16_t>( "waveform3", offsetof(FccdConfigV2_Data,waveform3) );
  confType.insert_native<uint16_t>( "waveform4", offsetof(FccdConfigV2_Data,waveform4) );
  confType.insert_native<uint16_t>( "waveform5", offsetof(FccdConfigV2_Data,waveform5) );
  confType.insert_native<uint16_t>( "waveform6", offsetof(FccdConfigV2_Data,waveform6) );
  confType.insert_native<uint16_t>( "waveform7", offsetof(FccdConfigV2_Data,waveform7) );
  confType.insert_native<uint16_t>( "waveform8", offsetof(FccdConfigV2_Data,waveform8) );
  confType.insert_native<uint16_t>( "waveform9", offsetof(FccdConfigV2_Data,waveform9) );
  confType.insert_native<uint16_t>( "waveform10", offsetof(FccdConfigV2_Data,waveform10) );
  confType.insert_native<uint16_t>( "waveform11", offsetof(FccdConfigV2_Data,waveform11) );
  confType.insert_native<uint16_t>( "waveform12", offsetof(FccdConfigV2_Data,waveform12) );
  confType.insert_native<uint16_t>( "waveform13", offsetof(FccdConfigV2_Data,waveform13) );
  confType.insert_native<uint16_t>( "waveform14", offsetof(FccdConfigV2_Data,waveform14) );

  return confType ;
}

void
FccdConfigV2::store( const Pds::FCCD::FccdConfigV2& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  FccdConfigV2 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
