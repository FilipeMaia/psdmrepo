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

class XtcClockTime  {
public:

  // Default constructor
  XtcClockTime () {}
  XtcClockTime ( const Pds::ClockTime& time ) ;
  XtcClockTime ( uint32_t sec, uint32_t nsec ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:

  uint32_t seconds;
  uint32_t nanoseconds;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_XTCCLOCKTIME_H
