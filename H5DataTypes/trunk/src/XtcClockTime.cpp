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
#include "Lusi/Lusi.h"

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
XtcClockTime::persType()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<XtcClockTime>() ;
  type.insert( "seconds", offsetof(XtcClockTime_Data,seconds), hdf5pp::AtomicType::atomicType<uint32_t>() ) ;
  type.insert( "nanoseconds", offsetof(XtcClockTime_Data,nanoseconds), hdf5pp::AtomicType::atomicType<uint32_t>() ) ;

  return type ;
}

} // namespace H5DataTypes
