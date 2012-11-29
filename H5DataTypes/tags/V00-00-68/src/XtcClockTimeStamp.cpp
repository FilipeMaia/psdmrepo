//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcClockTimeStamp...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/XtcClockTimeStamp.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

bool XtcClockTimeStamp::s_storeFullTimeStamp = true;

XtcClockTimeStamp::XtcClockTimeStamp ( const Pds::ClockTime& time, const Pds::TimeStamp& ts )
  : seconds(time.seconds())
  , nanoseconds(time.nanoseconds())
  , ticks(ts.ticks())
  , fiducials(ts.fiducials())
  , control(ts.control())
  , vector(ts.vector())
{
}

hdf5pp::Type
XtcClockTimeStamp::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
XtcClockTimeStamp::native_type()
{
  if (s_storeFullTimeStamp) {

    hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<XtcClockTimeStamp>() ;
    type.insert_native<uint32_t>( "seconds", offsetof(XtcClockTimeStamp,seconds) ) ;
    type.insert_native<uint32_t>( "nanoseconds", offsetof(XtcClockTimeStamp,nanoseconds) ) ;
    type.insert_native<uint32_t>( "ticks", offsetof(XtcClockTimeStamp, ticks) ) ;
    type.insert_native<uint32_t>( "fiducials", offsetof(XtcClockTimeStamp, fiducials) ) ;
    type.insert_native<uint32_t>( "control", offsetof(XtcClockTimeStamp, control) ) ;
    type.insert_native<uint32_t>( "vector", offsetof(XtcClockTimeStamp, vector) ) ;
    return type ;

  } else {

    hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType(2*sizeof(uint32_t));
    type.insert_native<uint32_t>( "seconds", offsetof(XtcClockTimeStamp,seconds) ) ;
    type.insert_native<uint32_t>( "nanoseconds", offsetof(XtcClockTimeStamp,nanoseconds) ) ;
    return type ;

  }
}

} // namespace H5DataTypes
