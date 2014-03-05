#ifndef H5DATATYPES_CONTROLDATACONFIGV3_H
#define H5DATATYPES_CONTROLDATACONFIGV3_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ControlDataConfigV3.
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
#include "H5DataTypes/XtcClockTime.h"
#include "hdf5pp/Group.h"
#include "pdsdata/psddl/control.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::ControlData::ConfigV3
//
class ControlDataConfigV3  {
public:

  typedef Pds::ControlData::ConfigV3 XtcType ;

  ControlDataConfigV3 () {}
  ControlDataConfigV3 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return xtc._sizeof() ; }

private:

  uint8_t   uses_l3t_events;
  uint8_t   uses_duration;
  uint8_t   uses_events;
  XtcClockTime duration;
  uint32_t  events;
  uint32_t  npvControls;
  uint32_t  npvMonitors;
  uint32_t  npvLabels;

};


} // namespace H5DataTypes

#endif // H5DATATYPES_CONTROLDATACONFIGV3_H
