#ifndef H5DATATYPES_XTCCLOCKTIME_H
#define H5DATATYPES_XTCCLOCKTIME_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcClockTime.
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

struct XtcClockTime_Data  {
  uint32_t seconds;
  uint32_t nanoseconds;
};


class XtcClockTime  {
public:

  // Default constructor
  XtcClockTime () {}
  XtcClockTime ( const Pds::ClockTime& time ) ;

  static hdf5pp::Type persType() ;

private:
  XtcClockTime_Data m_data ;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_XTCCLOCKTIME_H
