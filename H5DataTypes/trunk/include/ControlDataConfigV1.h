#ifndef H5DATATYPES_CONTROLDATACONFIGV1_H
#define H5DATATYPES_CONTROLDATACONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ControlDataConfigV1.
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
// Helper type for Pds::ControlData::PVControl
//
class ControlDataPVControlV1 {
public:

  ControlDataPVControlV1 () {}
  ControlDataPVControlV1 ( const Pds::ControlData::PVControl& pconfig ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:

  char     name[Pds::ControlData::PVControl::NameSize];
  uint32_t index;
  double   value;
};


//
// Helper type for Pds::ControlData::PVMonitor
//
class ControlDataPVMonitorV1 {
public:

  ControlDataPVMonitorV1 () {}
  ControlDataPVMonitorV1 ( const Pds::ControlData::PVMonitor& pconfig ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:

  char     name[Pds::ControlData::PVMonitor::NameSize];
  uint32_t index;
  double   loValue;
  double   hiValue;

};

//
// Helper type for Pds::ControlData::ConfigV1
//
class ControlDataConfigV1  {
public:

  typedef Pds::ControlData::ConfigV1 XtcType ;

  ControlDataConfigV1 () {}
  ControlDataConfigV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return xtc._sizeof() ; }

private:

  uint8_t   uses_duration;
  uint8_t   uses_events;
  XtcClockTime duration;
  uint32_t  events;
  uint32_t  npvControls;
  uint32_t  npvMonitors;

};


} // namespace H5DataTypes

#endif // H5DATATYPES_CONTROLDATACONFIGV1_H
