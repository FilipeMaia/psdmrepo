//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcClockTime...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/XtcClockTime.h"

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

XtcClockTime::XtcClockTime ( const Pds::ClockTime& time )
  : seconds(time.seconds())
  , nanoseconds(time.nanoseconds())
{
}

XtcClockTime::XtcClockTime ( uint32_t sec, uint32_t nsec )
  : seconds(sec)
  , nanoseconds(nsec)
{
}

hdf5pp::Type
XtcClockTime::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
XtcClockTime::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<XtcClockTime>() ;
  type.insert_native<uint32_t>( "seconds", offsetof(XtcClockTime,seconds) ) ;
  type.insert_native<uint32_t>( "nanoseconds", offsetof(XtcClockTime,nanoseconds) ) ;

  return type ;
}

} // namespace H5DataTypes
