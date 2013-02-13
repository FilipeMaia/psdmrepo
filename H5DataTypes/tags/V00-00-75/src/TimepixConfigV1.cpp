//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TimepixConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/TimepixConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "H5DataTypes/H5DataUtils.h"
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/EnumType.h"

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
TimepixConfigV1::TimepixConfigV1 ( const Pds::Timepix::ConfigV1& data )
  : readoutSpeed(data.readoutSpeed())
  , triggerMode(data.triggerMode())
  , shutterTimeout(data.shutterTimeout())
  , dac0Ikrum(data.dac0Ikrum())
  , dac0Disc(data.dac0Disc())
  , dac0Preamp(data.dac0Preamp())
  , dac0BufAnalogA(data.dac0BufAnalogA())
  , dac0BufAnalogB(data.dac0BufAnalogB())
  , dac0Hist(data.dac0Hist())
  , dac0ThlFine(data.dac0ThlFine())
  , dac0ThlCourse(data.dac0ThlCourse())
  , dac0Vcas(data.dac0Vcas())
  , dac0Fbk(data.dac0Fbk())
  , dac0Gnd(data.dac0Gnd())
  , dac0Ths(data.dac0Ths())
  , dac0BiasLvds(data.dac0BiasLvds())
  , dac0RefLvds(data.dac0RefLvds())
  , dac1Ikrum(data.dac1Ikrum())
  , dac1Disc(data.dac1Disc())
  , dac1Preamp(data.dac1Preamp())
  , dac1BufAnalogA(data.dac1BufAnalogA())
  , dac1BufAnalogB(data.dac1BufAnalogB())
  , dac1Hist(data.dac1Hist())
  , dac1ThlFine(data.dac1ThlFine())
  , dac1ThlCourse(data.dac1ThlCourse())
  , dac1Vcas(data.dac1Vcas())
  , dac1Fbk(data.dac1Fbk())
  , dac1Gnd(data.dac1Gnd())
  , dac1Ths(data.dac1Ths())
  , dac1BiasLvds(data.dac1BiasLvds())
  , dac1RefLvds(data.dac1RefLvds())
  , dac2Ikrum(data.dac2Ikrum())
  , dac2Disc(data.dac2Disc())
  , dac2Preamp(data.dac2Preamp())
  , dac2BufAnalogA(data.dac2BufAnalogA())
  , dac2BufAnalogB(data.dac2BufAnalogB())
  , dac2Hist(data.dac2Hist())
  , dac2ThlFine(data.dac2ThlFine())
  , dac2ThlCourse(data.dac2ThlCourse())
  , dac2Vcas(data.dac2Vcas())
  , dac2Fbk(data.dac2Fbk())
  , dac2Gnd(data.dac2Gnd())
  , dac2Ths(data.dac2Ths())
  , dac2BiasLvds(data.dac2BiasLvds())
  , dac2RefLvds(data.dac2RefLvds())
  , dac3Ikrum(data.dac3Ikrum())
  , dac3Disc(data.dac3Disc())
  , dac3Preamp(data.dac3Preamp())
  , dac3BufAnalogA(data.dac3BufAnalogA())
  , dac3BufAnalogB(data.dac3BufAnalogB())
  , dac3Hist(data.dac3Hist())
  , dac3ThlFine(data.dac3ThlFine())
  , dac3ThlCourse(data.dac3ThlCourse())
  , dac3Vcas(data.dac3Vcas())
  , dac3Fbk(data.dac3Fbk())
  , dac3Gnd(data.dac3Gnd())
  , dac3Ths(data.dac3Ths())
  , dac3BiasLvds(data.dac3BiasLvds())
  , dac3RefLvds(data.dac3RefLvds())
{
}

hdf5pp::Type
TimepixConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
TimepixConfigV1::native_type()
{
  hdf5pp::EnumType<uint8_t> speedEnum = hdf5pp::EnumType<uint8_t>::enumType() ;
  speedEnum.insert ( "ReadoutSpeed_Slow", Pds::Timepix::ConfigV1::ReadoutSpeed_Slow ) ;
  speedEnum.insert ( "ReadoutSpeed_Fast", Pds::Timepix::ConfigV1::ReadoutSpeed_Fast ) ;

  hdf5pp::EnumType<uint8_t> trigEnum = hdf5pp::EnumType<uint8_t>::enumType() ;
  trigEnum.insert ( "TriggerMode_ExtPos", Pds::Timepix::ConfigV1::TriggerMode_ExtPos ) ;
  trigEnum.insert ( "TriggerMode_ExtNeg", Pds::Timepix::ConfigV1::TriggerMode_ExtNeg ) ;
  trigEnum.insert ( "TriggerMode_Soft", Pds::Timepix::ConfigV1::TriggerMode_Soft ) ;

  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<TimepixConfigV1>() ;
  confType.insert("readoutSpeed", offsetof(TimepixConfigV1, readoutSpeed), speedEnum);
  confType.insert("triggerMode", offsetof(TimepixConfigV1, triggerMode), trigEnum);
  confType.insert_native<int32_t>("shutterTimeout", offsetof(TimepixConfigV1, shutterTimeout));
  confType.insert_native<int32_t>("dac0Ikrum", offsetof(TimepixConfigV1, dac0Ikrum));
  confType.insert_native<int32_t>("dac0Disc", offsetof(TimepixConfigV1, dac0Disc));
  confType.insert_native<int32_t>("dac0Preamp", offsetof(TimepixConfigV1, dac0Preamp));
  confType.insert_native<int32_t>("dac0BufAnalogA", offsetof(TimepixConfigV1, dac0BufAnalogA));
  confType.insert_native<int32_t>("dac0BufAnalogB", offsetof(TimepixConfigV1, dac0BufAnalogB));
  confType.insert_native<int32_t>("dac0Hist", offsetof(TimepixConfigV1, dac0Hist));
  confType.insert_native<int32_t>("dac0ThlFine", offsetof(TimepixConfigV1, dac0ThlFine));
  confType.insert_native<int32_t>("dac0ThlCourse", offsetof(TimepixConfigV1, dac0ThlCourse));
  confType.insert_native<int32_t>("dac0Vcas", offsetof(TimepixConfigV1, dac0Vcas));
  confType.insert_native<int32_t>("dac0Fbk", offsetof(TimepixConfigV1, dac0Fbk));
  confType.insert_native<int32_t>("dac0Gnd", offsetof(TimepixConfigV1, dac0Gnd));
  confType.insert_native<int32_t>("dac0Ths", offsetof(TimepixConfigV1, dac0Ths));
  confType.insert_native<int32_t>("dac0BiasLvds", offsetof(TimepixConfigV1, dac0BiasLvds));
  confType.insert_native<int32_t>("dac0RefLvds", offsetof(TimepixConfigV1, dac0RefLvds));
  confType.insert_native<int32_t>("dac1Ikrum", offsetof(TimepixConfigV1, dac1Ikrum));
  confType.insert_native<int32_t>("dac1Disc", offsetof(TimepixConfigV1, dac1Disc));
  confType.insert_native<int32_t>("dac1Preamp", offsetof(TimepixConfigV1, dac1Preamp));
  confType.insert_native<int32_t>("dac1BufAnalogA", offsetof(TimepixConfigV1, dac1BufAnalogA));
  confType.insert_native<int32_t>("dac1BufAnalogB", offsetof(TimepixConfigV1, dac1BufAnalogB));
  confType.insert_native<int32_t>("dac1Hist", offsetof(TimepixConfigV1, dac1Hist));
  confType.insert_native<int32_t>("dac1ThlFine", offsetof(TimepixConfigV1, dac1ThlFine));
  confType.insert_native<int32_t>("dac1ThlCourse", offsetof(TimepixConfigV1, dac1ThlCourse));
  confType.insert_native<int32_t>("dac1Vcas", offsetof(TimepixConfigV1, dac1Vcas));
  confType.insert_native<int32_t>("dac1Fbk", offsetof(TimepixConfigV1, dac1Fbk));
  confType.insert_native<int32_t>("dac1Gnd", offsetof(TimepixConfigV1, dac1Gnd));
  confType.insert_native<int32_t>("dac1Ths", offsetof(TimepixConfigV1, dac1Ths));
  confType.insert_native<int32_t>("dac1BiasLvds", offsetof(TimepixConfigV1, dac1BiasLvds));
  confType.insert_native<int32_t>("dac1RefLvds", offsetof(TimepixConfigV1, dac1RefLvds));
  confType.insert_native<int32_t>("dac2Ikrum", offsetof(TimepixConfigV1, dac2Ikrum));
  confType.insert_native<int32_t>("dac2Disc", offsetof(TimepixConfigV1, dac2Disc));
  confType.insert_native<int32_t>("dac2Preamp", offsetof(TimepixConfigV1, dac2Preamp));
  confType.insert_native<int32_t>("dac2BufAnalogA", offsetof(TimepixConfigV1, dac2BufAnalogA));
  confType.insert_native<int32_t>("dac2BufAnalogB", offsetof(TimepixConfigV1, dac2BufAnalogB));
  confType.insert_native<int32_t>("dac2Hist", offsetof(TimepixConfigV1, dac2Hist));
  confType.insert_native<int32_t>("dac2ThlFine", offsetof(TimepixConfigV1, dac2ThlFine));
  confType.insert_native<int32_t>("dac2ThlCourse", offsetof(TimepixConfigV1, dac2ThlCourse));
  confType.insert_native<int32_t>("dac2Vcas", offsetof(TimepixConfigV1, dac2Vcas));
  confType.insert_native<int32_t>("dac2Fbk", offsetof(TimepixConfigV1, dac2Fbk));
  confType.insert_native<int32_t>("dac2Gnd", offsetof(TimepixConfigV1, dac2Gnd));
  confType.insert_native<int32_t>("dac2Ths", offsetof(TimepixConfigV1, dac2Ths));
  confType.insert_native<int32_t>("dac2BiasLvds", offsetof(TimepixConfigV1, dac2BiasLvds));
  confType.insert_native<int32_t>("dac2RefLvds", offsetof(TimepixConfigV1, dac2RefLvds));
  confType.insert_native<int32_t>("dac3Ikrum", offsetof(TimepixConfigV1, dac3Ikrum));
  confType.insert_native<int32_t>("dac3Disc", offsetof(TimepixConfigV1, dac3Disc));
  confType.insert_native<int32_t>("dac3Preamp", offsetof(TimepixConfigV1, dac3Preamp));
  confType.insert_native<int32_t>("dac3BufAnalogA", offsetof(TimepixConfigV1, dac3BufAnalogA));
  confType.insert_native<int32_t>("dac3BufAnalogB", offsetof(TimepixConfigV1, dac3BufAnalogB));
  confType.insert_native<int32_t>("dac3Hist", offsetof(TimepixConfigV1, dac3Hist));
  confType.insert_native<int32_t>("dac3ThlFine", offsetof(TimepixConfigV1, dac3ThlFine));
  confType.insert_native<int32_t>("dac3ThlCourse", offsetof(TimepixConfigV1, dac3ThlCourse));
  confType.insert_native<int32_t>("dac3Vcas", offsetof(TimepixConfigV1, dac3Vcas));
  confType.insert_native<int32_t>("dac3Fbk", offsetof(TimepixConfigV1, dac3Fbk));
  confType.insert_native<int32_t>("dac3Gnd", offsetof(TimepixConfigV1, dac3Gnd));
  confType.insert_native<int32_t>("dac3Ths", offsetof(TimepixConfigV1, dac3Ths));
  confType.insert_native<int32_t>("dac3BiasLvds", offsetof(TimepixConfigV1, dac3BiasLvds));
  confType.insert_native<int32_t>("dac3RefLvds", offsetof(TimepixConfigV1, dac3RefLvds));

  return confType ;
}

void
TimepixConfigV1::store( const Pds::Timepix::ConfigV1& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  TimepixConfigV1 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}


} // namespace H5DataTypes
