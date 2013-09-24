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
  : width(data.width())
  , height(data.height())
  , trimmedWidth(data.trimmedWidth())
  , trimmedHeight(data.trimmedHeight())
  , outputMode(data.outputMode())
  , ccdEnable(data.ccdEnable())
  , focusMode(data.focusMode())
  , exposureTime(data.exposureTime())
{
  const ndarray<const uint16_t, 1>& waveforms = data.waveforms();
  waveform0 = waveforms[0];
  waveform1 = waveforms[1];
  waveform2 = waveforms[2];
  waveform3 = waveforms[3];
  waveform4 = waveforms[4];
  waveform5 = waveforms[5];
  waveform6 = waveforms[6];
  waveform7 = waveforms[7];
  waveform8 = waveforms[8];
  waveform9 = waveforms[9];
  waveform10 = waveforms[10];
  waveform11 = waveforms[11];
  waveform12 = waveforms[12];
  waveform13 = waveforms[13];
  waveform14 = waveforms[14];

  const ndarray<const float, 1>& dacVoltages = data.dacVoltages();
  dacVoltage1 = dacVoltages[0];
  dacVoltage2 = dacVoltages[1];
  dacVoltage3 = dacVoltages[2];
  dacVoltage4 = dacVoltages[3];
  dacVoltage5 = dacVoltages[4];
  dacVoltage6 = dacVoltages[5];
  dacVoltage7 = dacVoltages[6];
  dacVoltage8 = dacVoltages[7];
  dacVoltage9 = dacVoltages[8];
  dacVoltage10 = dacVoltages[9];
  dacVoltage11 = dacVoltages[10];
  dacVoltage12 = dacVoltages[11];
  dacVoltage13 = dacVoltages[12];
  dacVoltage14 = dacVoltages[13];
  dacVoltage15 = dacVoltages[14];
  dacVoltage16 = dacVoltages[15];
  dacVoltage17 = dacVoltages[16];
}

hdf5pp::Type
FccdConfigV2::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
FccdConfigV2::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<FccdConfigV2>() ;
  confType.insert_native<uint32_t>( "width", offsetof(FccdConfigV2,width) );
  confType.insert_native<uint32_t>( "height", offsetof(FccdConfigV2,height) );
  confType.insert_native<uint32_t>( "trimmedWidth", offsetof(FccdConfigV2,trimmedWidth) );
  confType.insert_native<uint32_t>( "trimmedHeight", offsetof(FccdConfigV2,trimmedHeight) );
  confType.insert_native<uint16_t>( "outputMode", offsetof(FccdConfigV2,outputMode) );
  confType.insert_native<uint8_t>( "ccdEnable", offsetof(FccdConfigV2,ccdEnable) );
  confType.insert_native<uint8_t>( "focusMode", offsetof(FccdConfigV2,focusMode) );
  confType.insert_native<uint32_t>( "exposureTime", offsetof(FccdConfigV2,exposureTime) );
  confType.insert_native<float>( "dacVoltage1", offsetof(FccdConfigV2,dacVoltage1) );
  confType.insert_native<float>( "dacVoltage2", offsetof(FccdConfigV2,dacVoltage2) );
  confType.insert_native<float>( "dacVoltage3", offsetof(FccdConfigV2,dacVoltage3) );
  confType.insert_native<float>( "dacVoltage4", offsetof(FccdConfigV2,dacVoltage4) );
  confType.insert_native<float>( "dacVoltage5", offsetof(FccdConfigV2,dacVoltage5) );
  confType.insert_native<float>( "dacVoltage6", offsetof(FccdConfigV2,dacVoltage6) );
  confType.insert_native<float>( "dacVoltage7", offsetof(FccdConfigV2,dacVoltage7) );
  confType.insert_native<float>( "dacVoltage8", offsetof(FccdConfigV2,dacVoltage8) );
  confType.insert_native<float>( "dacVoltage9", offsetof(FccdConfigV2,dacVoltage9) );
  confType.insert_native<float>( "dacVoltage10", offsetof(FccdConfigV2,dacVoltage10) );
  confType.insert_native<float>( "dacVoltage11", offsetof(FccdConfigV2,dacVoltage11) );
  confType.insert_native<float>( "dacVoltage12", offsetof(FccdConfigV2,dacVoltage12) );
  confType.insert_native<float>( "dacVoltage13", offsetof(FccdConfigV2,dacVoltage13) );
  confType.insert_native<float>( "dacVoltage14", offsetof(FccdConfigV2,dacVoltage14) );
  confType.insert_native<float>( "dacVoltage15", offsetof(FccdConfigV2,dacVoltage15) );
  confType.insert_native<float>( "dacVoltage16", offsetof(FccdConfigV2,dacVoltage16) );
  confType.insert_native<float>( "dacVoltage17", offsetof(FccdConfigV2,dacVoltage17) );
  confType.insert_native<uint16_t>( "waveform0", offsetof(FccdConfigV2,waveform0) );
  confType.insert_native<uint16_t>( "waveform1", offsetof(FccdConfigV2,waveform1) );
  confType.insert_native<uint16_t>( "waveform2", offsetof(FccdConfigV2,waveform2) );
  confType.insert_native<uint16_t>( "waveform3", offsetof(FccdConfigV2,waveform3) );
  confType.insert_native<uint16_t>( "waveform4", offsetof(FccdConfigV2,waveform4) );
  confType.insert_native<uint16_t>( "waveform5", offsetof(FccdConfigV2,waveform5) );
  confType.insert_native<uint16_t>( "waveform6", offsetof(FccdConfigV2,waveform6) );
  confType.insert_native<uint16_t>( "waveform7", offsetof(FccdConfigV2,waveform7) );
  confType.insert_native<uint16_t>( "waveform8", offsetof(FccdConfigV2,waveform8) );
  confType.insert_native<uint16_t>( "waveform9", offsetof(FccdConfigV2,waveform9) );
  confType.insert_native<uint16_t>( "waveform10", offsetof(FccdConfigV2,waveform10) );
  confType.insert_native<uint16_t>( "waveform11", offsetof(FccdConfigV2,waveform11) );
  confType.insert_native<uint16_t>( "waveform12", offsetof(FccdConfigV2,waveform12) );
  confType.insert_native<uint16_t>( "waveform13", offsetof(FccdConfigV2,waveform13) );
  confType.insert_native<uint16_t>( "waveform14", offsetof(FccdConfigV2,waveform14) );

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
