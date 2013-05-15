#ifndef H5DATATYPES_TIMEPIXCONFIGV2_H
#define H5DATATYPES_TIMEPIXCONFIGV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TimepixConfigV2.
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
#include "pdsdata/timepix/ConfigV2.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Timepix::ConfigV2
//
class TimepixConfigV2  {
public:

  typedef Pds::Timepix::ConfigV2 XtcType ;

  TimepixConfigV2 () {}
  TimepixConfigV2 ( const XtcType& config ) ;

  ~TimepixConfigV2();

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static void store ( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof xtc ; }

private:

  uint8_t readoutSpeed;
  uint8_t triggerMode;
  int32_t timepixSpeed;
  int32_t dac0Ikrum;
  int32_t dac0Disc;
  int32_t dac0Preamp;
  int32_t dac0BufAnalogA;
  int32_t dac0BufAnalogB;
  int32_t dac0Hist;
  int32_t dac0ThlFine;
  int32_t dac0ThlCourse;
  int32_t dac0Vcas;
  int32_t dac0Fbk;
  int32_t dac0Gnd;
  int32_t dac0Ths;
  int32_t dac0BiasLvds;
  int32_t dac0RefLvds;
  int32_t dac1Ikrum;
  int32_t dac1Disc;
  int32_t dac1Preamp;
  int32_t dac1BufAnalogA;
  int32_t dac1BufAnalogB;
  int32_t dac1Hist;
  int32_t dac1ThlFine;
  int32_t dac1ThlCourse;
  int32_t dac1Vcas;
  int32_t dac1Fbk;
  int32_t dac1Gnd;
  int32_t dac1Ths;
  int32_t dac1BiasLvds;
  int32_t dac1RefLvds;
  int32_t dac2Ikrum;
  int32_t dac2Disc;
  int32_t dac2Preamp;
  int32_t dac2BufAnalogA;
  int32_t dac2BufAnalogB;
  int32_t dac2Hist;
  int32_t dac2ThlFine;
  int32_t dac2ThlCourse;
  int32_t dac2Vcas;
  int32_t dac2Fbk;
  int32_t dac2Gnd;
  int32_t dac2Ths;
  int32_t dac2BiasLvds;
  int32_t dac2RefLvds;
  int32_t dac3Ikrum;
  int32_t dac3Disc;
  int32_t dac3Preamp;
  int32_t dac3BufAnalogA;
  int32_t dac3BufAnalogB;
  int32_t dac3Hist;
  int32_t dac3ThlFine;
  int32_t dac3ThlCourse;
  int32_t dac3Vcas;
  int32_t dac3Fbk;
  int32_t dac3Gnd;
  int32_t dac3Ths;
  int32_t dac3BiasLvds;
  int32_t dac3RefLvds;
  int32_t chipCount;
  int32_t driverVersion;
  uint32_t firmwareVersion;
  uint32_t pixelThreshSize;
  uint8_t pixelThresh[XtcType::PixelThreshMax];
  char*   chip0Name;
  char*   chip1Name;
  char*   chip2Name;
  char*   chip3Name;
  int32_t chip0ID;
  int32_t chip1ID;
  int32_t chip2ID;
  int32_t chip3ID;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_TIMEPIXCONFIGV2_H
