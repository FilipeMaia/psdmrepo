#ifndef H5DATATYPES_XTCCLOCKTIMESTAMP_H
#define H5DATATYPES_XTCCLOCKTIMESTAMP_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcClockTimeStamp, combination of Clocktime and Timestamp data
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
#include "pdsdata/xtc/ClockTime.hh"
#include "pdsdata/xtc/TimeStamp.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace H5DataTypes {

class XtcClockTimeStamp  {
public:

  // Default constructor
  XtcClockTimeStamp () {}
  XtcClockTimeStamp ( const Pds::ClockTime& time, const Pds::TimeStamp& ts ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static void storeFullTimeStamp(bool val) { s_storeFullTimeStamp = val; }

private:

  uint32_t seconds;
  uint32_t nanoseconds;
  uint32_t ticks;
  uint32_t fiducials;
  uint32_t control;
  uint32_t vector;

  static bool s_storeFullTimeStamp;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_XTCCLOCKTIMESTAMP_H
