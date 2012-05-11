#ifndef H5DATATYPES_FCCDCONFIGV2_H
#define H5DATATYPES_FCCDCONFIGV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FccdConfigV2.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------


//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Group.h"
#include "pdsdata/fccd/FccdConfigV2.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::FCCD::FccdConfigV2
//
struct FccdConfigV2_Data  {
  uint32_t width;
  uint32_t height;
  uint32_t trimmedWidth;
  uint32_t trimmedHeight;
  uint16_t outputMode;
  uint8_t  ccdEnable;
  uint8_t  focusMode;
  uint32_t exposureTime;
  float    dacVoltage1;
  float    dacVoltage2;
  float    dacVoltage3;
  float    dacVoltage4;
  float    dacVoltage5;
  float    dacVoltage6;
  float    dacVoltage7;
  float    dacVoltage8;
  float    dacVoltage9;
  float    dacVoltage10;
  float    dacVoltage11;
  float    dacVoltage12;
  float    dacVoltage13;
  float    dacVoltage14;
  float    dacVoltage15;
  float    dacVoltage16;
  float    dacVoltage17;
  uint16_t waveform0;
  uint16_t waveform1;
  uint16_t waveform2;
  uint16_t waveform3;
  uint16_t waveform4;
  uint16_t waveform5;
  uint16_t waveform6;
  uint16_t waveform7;
  uint16_t waveform8;
  uint16_t waveform9;
  uint16_t waveform10;
  uint16_t waveform11;
  uint16_t waveform12;
  uint16_t waveform13;
  uint16_t waveform14;
};

class FccdConfigV2  {
public:

  typedef Pds::FCCD::FccdConfigV2 XtcType ;

  FccdConfigV2 () {}
  FccdConfigV2 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

private:

  FccdConfigV2_Data m_data ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_FCCDCONFIGV2_H
