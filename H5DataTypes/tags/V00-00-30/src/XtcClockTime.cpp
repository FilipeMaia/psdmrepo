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
#include "SITConfig/SITConfig.h"

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
{
  m_data.seconds = time.seconds() ;
  m_data.nanoseconds = time.nanoseconds() ;
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
  type.insert_native<uint32_t>( "seconds", offsetof(XtcClockTime_Data,seconds) ) ;
  type.insert_native<uint32_t>( "nanoseconds", offsetof(XtcClockTime_Data,nanoseconds) ) ;

  return type ;
}

} // namespace H5DataTypes
